using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

// 완성된 QA CSV의 품질을 분석하고 검색 실험용 표본을 만드는 처리기입니다.
// C#은 정적 타입 언어이므로 모든 입력과 반환값의 타입이 코드에 명시되어 있습니다.
public static class SuperExperimentProcessor
{
    private sealed class Qa
    {
        public string DocumentId = "", Category = "", ConversationId = "", QaNumber = "";
        public string Intent = "", Question = "", Answer = "";
        public int QuestionLength, AnswerLength;
    }

    // 따옴표 안의 쉼표, 따옴표 두 개, 줄바꿈을 모두 처리하는 CSV 판독기입니다.
    private static IEnumerable<string[]> ReadCsv(TextReader reader)
    {
        var row = new List<string>();
        var field = new StringBuilder();
        bool quoted = false;
        while (true)
        {
            int value = reader.Read();
            if (value < 0)
            {
                if (quoted) throw new InvalidDataException("따옴표가 닫히지 않은 CSV입니다.");
                if (field.Length > 0 || row.Count > 0) { row.Add(field.ToString()); yield return row.ToArray(); }
                yield break;
            }
            char ch = (char)value;
            if (quoted)
            {
                if (ch == '"')
                {
                    if (reader.Peek() == '"') { reader.Read(); field.Append('"'); }
                    else quoted = false;
                }
                else field.Append(ch);
            }
            else if (ch == '"' && field.Length == 0) quoted = true;
            else if (ch == ',') { row.Add(field.ToString()); field.Clear(); }
            else if (ch == '\r' || ch == '\n')
            {
                if (ch == '\r' && reader.Peek() == '\n') reader.Read();
                row.Add(field.ToString()); field.Clear(); yield return row.ToArray(); row.Clear();
            }
            else field.Append(ch);
        }
    }

    private static List<Qa> LoadQa(string path)
    {
        var result = new List<Qa>();
        using (var reader = new StreamReader(path, new UTF8Encoding(true, true), true))
        using (var rows = ReadCsv(reader).GetEnumerator())
        {
            if (!rows.MoveNext()) throw new InvalidDataException("QA CSV가 비어 있습니다.");
            string[] header = rows.Current;
            var idx = header.Select((name, i) => new { name, i }).ToDictionary(x => x.name, x => x.i);
            string[] required = { "document_id", "category", "conversation_id", "qa_number", "intent", "question", "answer", "question_length", "answer_length" };
            foreach (string name in required) if (!idx.ContainsKey(name)) throw new InvalidDataException("필수 컬럼이 없습니다: " + name);
            while (rows.MoveNext())
            {
                string[] r = rows.Current;
                int qLength, aLength;
                if (!int.TryParse(r[idx["question_length"]], out qLength)) qLength = r[idx["question"]].Length;
                if (!int.TryParse(r[idx["answer_length"]], out aLength)) aLength = r[idx["answer"]].Length;
                result.Add(new Qa {
                    DocumentId = r[idx["document_id"]], Category = r[idx["category"]], ConversationId = r[idx["conversation_id"]],
                    QaNumber = r[idx["qa_number"]], Intent = r[idx["intent"]], Question = r[idx["question"]], Answer = r[idx["answer"]],
                    QuestionLength = qLength, AnswerLength = aLength
                });
            }
        }
        return result;
    }

    private static string Csv(string value) { return "\"" + (value ?? "").Replace("\"", "\"\"") + "\""; }
    private static string Md(string value, int max)
    {
        string clean = (value ?? "").Replace("\r", " ").Replace("\n", " ").Replace("|", "\\|");
        return clean.Length <= max ? clean : clean.Substring(0, max) + "…";
    }
    private static double Median(IEnumerable<int> source)
    {
        int[] values = source.OrderBy(x => x).ToArray();
        if (values.Length == 0) return 0;
        int middle = values.Length / 2;
        return values.Length % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
    }
    private static string LengthBucket(int length)
    {
        if (length <= 4) return "1~4자";
        if (length <= 10) return "5~10자";
        if (length <= 20) return "11~20자";
        if (length <= 40) return "21~40자";
        return "41자 이상";
    }
    private static void Shuffle<T>(List<T> list, Random random)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1); T temp = list[i]; list[i] = list[j]; list[j] = temp;
        }
    }

    // 인텐트별로 한 건씩 순환하며 선택하여 대형 인텐트의 독점을 방지합니다.
    private static List<Qa> StratifiedSample(List<Qa> candidates, int target, int maxPerIntent, int seed)
    {
        var groups = candidates.GroupBy(x => x.Intent).OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(x => x.Key, x => x.ToList(), StringComparer.Ordinal);
        var random = new Random(seed);
        foreach (string key in groups.Keys.OrderBy(x => x, StringComparer.Ordinal).ToArray()) Shuffle(groups[key], random);
        var taken = groups.Keys.ToDictionary(x => x, x => 0, StringComparer.Ordinal);
        var sample = new List<Qa>(target);
        bool progressed = true;
        while (sample.Count < target && progressed)
        {
            progressed = false;
            foreach (string intent in groups.Keys.OrderBy(x => x, StringComparer.Ordinal))
            {
                int index = taken[intent];
                if (index < groups[intent].Count && index < maxPerIntent && sample.Count < target)
                {
                    sample.Add(groups[intent][index]); taken[intent] = index + 1; progressed = true;
                }
            }
        }
        return sample;
    }

    private static void WriteQa(string path, IEnumerable<Qa> rows)
    {
        using (var writer = new StreamWriter(path, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("\"document_id\",\"category\",\"conversation_id\",\"qa_number\",\"intent\",\"question\",\"answer\",\"question_length\",\"answer_length\"");
            foreach (Qa q in rows) writer.WriteLine(string.Join(",", new[] { Csv(q.DocumentId), Csv(q.Category), Csv(q.ConversationId), Csv(q.QaNumber), Csv(q.Intent), Csv(q.Question), Csv(q.Answer), Csv(q.QuestionLength.ToString()), Csv(q.AnswerLength.ToString()) }));
        }
    }

    public static void Run(string trainPath, string validationPath, string root)
    {
        List<Qa> train = LoadQa(trainPath), validation = LoadQa(validationPath);
        int total = train.Count, emptyIntent = train.Count(x => string.IsNullOrWhiteSpace(x.Intent));
        var intentStats = train.GroupBy(x => string.IsNullOrWhiteSpace(x.Intent) ? "(빈 인텐트)" : x.Intent)
            .Select(x => new { Intent = x.Key, Count = x.Count(), Ratio = 100.0 * x.Count() / total })
            .OrderByDescending(x => x.Count).ThenBy(x => x.Intent, StringComparer.Ordinal).ToList();
        var questionGroups = train.GroupBy(x => x.Question, StringComparer.Ordinal).ToList();
        int uniqueQuestions = questionGroups.Count;
        int duplicateQuestionExcess = total - uniqueQuestions;
        var exactGroups = train.GroupBy(x => x.Question + "\u001f" + x.Answer, StringComparer.Ordinal).ToList();
        int exactDuplicateExcess = exactGroups.Sum(x => Math.Max(0, x.Count() - 1));
        var differentAnswerGroups = questionGroups.Where(x => x.Select(y => y.Answer).Distinct(StringComparer.Ordinal).Count() > 1).ToList();

        string results = Path.Combine(root, "results"), docs = Path.Combine(root, "docs"), experiment = Path.Combine(root, "data", "experiment");
        Directory.CreateDirectory(results); Directory.CreateDirectory(docs); Directory.CreateDirectory(experiment);

        using (var w = new StreamWriter(Path.Combine(results, "super_intent_distribution.csv"), false, new UTF8Encoding(true)))
        {
            w.WriteLine("\"intent\",\"qa_count\",\"ratio_percent\"");
            foreach (var x in intentStats) w.WriteLine(Csv(x.Intent) + "," + Csv(x.Count.ToString()) + "," + Csv(x.Ratio.ToString("0.0000", CultureInfo.InvariantCulture)));
        }
        using (var w = new StreamWriter(Path.Combine(results, "super_duplicate_questions.csv"), false, new UTF8Encoding(true)))
        {
            w.WriteLine("\"question\",\"qa_count\",\"duplicate_excess_count\",\"distinct_answer_count\",\"representative_answers\"");
            foreach (var g in questionGroups.Where(x => x.Count() > 1).OrderByDescending(x => x.Count()).ThenBy(x => x.Key, StringComparer.Ordinal))
            {
                string answers = string.Join(" || ", g.Select(x => x.Answer).Distinct(StringComparer.Ordinal).Take(5));
                w.WriteLine(string.Join(",", new[] { Csv(g.Key), Csv(g.Count().ToString()), Csv((g.Count() - 1).ToString()), Csv(g.Select(x => x.Answer).Distinct(StringComparer.Ordinal).Count().ToString()), Csv(answers) }));
            }
        }
        List<Qa> shortQuestions = train.Where(x => x.QuestionLength <= 4).ToList();
        WriteQa(Path.Combine(results, "super_short_questions.csv"), shortQuestions);

        // 빈 인텐트를 제외하고 완전 동일 QA를 document_id 오름차순의 첫 행 하나로 줄입니다.
        List<Qa> deduplicated = train.Where(x => !string.IsNullOrWhiteSpace(x.Intent)).OrderBy(x => x.DocumentId, StringComparer.Ordinal)
            .GroupBy(x => x.Question + "\u001f" + x.Answer, StringComparer.Ordinal).Select(x => x.First()).ToList();
        List<Qa> shortCandidates = deduplicated.Where(x => x.QuestionLength <= 4).ToList();
        List<Qa> trainCandidates = deduplicated.Where(x => x.QuestionLength > 4).ToList();
        List<Qa> trainSample = StratifiedSample(trainCandidates, 5000, 100, 42);
        WriteQa(Path.Combine(experiment, "super_train_sample_5000.csv"), trainSample);
        WriteQa(Path.Combine(experiment, "super_train_short_question_candidates.csv"), shortCandidates);

        using (var w = new StreamWriter(Path.Combine(results, "super_train_sample_distribution.csv"), false, new UTF8Encoding(true)))
        {
            w.WriteLine("\"intent\",\"eligible_count\",\"sample_count\",\"sample_ratio_percent\"");
            var selected = trainSample.GroupBy(x => x.Intent).ToDictionary(x => x.Key, x => x.Count(), StringComparer.Ordinal);
            foreach (var g in trainCandidates.GroupBy(x => x.Intent).OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                int count = selected.ContainsKey(g.Key) ? selected[g.Key] : 0;
                w.WriteLine(string.Join(",", new[] { Csv(g.Key), Csv(g.Count().ToString()), Csv(count.ToString()), Csv((100.0 * count / g.Count()).ToString("0.0000", CultureInfo.InvariantCulture)) }));
            }
        }

        var trainIntents = new HashSet<string>(trainSample.Select(x => x.Intent), StringComparer.Ordinal);
        List<Qa> validationCandidates = validation.Where(x => trainIntents.Contains(x.Intent) && x.QuestionLength > 4).ToList();
        List<Qa> validationSample = StratifiedSample(validationCandidates, 500, 20, 42);
        validationSample = validationSample.OrderBy(x => x.Intent, StringComparer.Ordinal).ThenBy(x => x.DocumentId, StringComparer.Ordinal).ToList();
        using (var w = new StreamWriter(Path.Combine(experiment, "super_validation_sample_500.csv"), false, new UTF8Encoding(true)))
        {
            w.WriteLine("\"query_id\",\"question\",\"reference_answer\",\"expected_intent\",\"category\",\"original_document_id\",\"conversation_id\",\"qa_number\"");
            for (int i = 0; i < validationSample.Count; i++)
            {
                Qa q = validationSample[i]; string queryId = "super_val_query_" + (i + 1).ToString("D6");
                w.WriteLine(string.Join(",", new[] { Csv(queryId), Csv(q.Question), Csv(q.Answer), Csv(q.Intent), Csv(q.Category), Csv(q.DocumentId), Csv(q.ConversationId), Csv(q.QaNumber) }));
            }
        }
        using (var w = new StreamWriter(Path.Combine(results, "super_validation_sample_distribution.csv"), false, new UTF8Encoding(true)))
        {
            w.WriteLine("\"intent\",\"eligible_count\",\"sample_count\",\"sample_ratio_percent\"");
            var selected = validationSample.GroupBy(x => x.Intent).ToDictionary(x => x.Key, x => x.Count(), StringComparer.Ordinal);
            foreach (var g in validationCandidates.GroupBy(x => x.Intent).OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                int count = selected.ContainsKey(g.Key) ? selected[g.Key] : 0;
                w.WriteLine(string.Join(",", new[] { Csv(g.Key), Csv(g.Count().ToString()), Csv(count.ToString()), Csv((100.0 * count / g.Count()).ToString("0.0000", CultureInfo.InvariantCulture)) }));
            }
        }

        int qMin = train.Min(x => x.QuestionLength), qMax = train.Max(x => x.QuestionLength);
        int aMin = train.Min(x => x.AnswerLength), aMax = train.Max(x => x.AnswerLength);
        double qAvg = train.Average(x => x.QuestionLength), aAvg = train.Average(x => x.AnswerLength);
        var qBuckets = train.GroupBy(x => LengthBucket(x.QuestionLength)).ToDictionary(x => x.Key, x => x.Count());
        var aBuckets = train.GroupBy(x => LengthBucket(x.AnswerLength)).ToDictionary(x => x.Key, x => x.Count());
        string[] bucketOrder = { "1~4자", "5~10자", "11~20자", "21~40자", "41자 이상" };

        var md = new StringBuilder();
        md.AppendLine("# 슈퍼 Training 완전 QA 데이터 품질 분석\n");
        md.AppendLine("분석 대상: `data/processed/슈퍼_train_qa_complete.csv`\n");
        md.AppendLine("## 핵심 통계\n");
        md.AppendLine("| 항목 | 값 |\n|---|---:|");
        md.AppendLine("| 전체 QA 수 | " + total + " |");
        md.AppendLine("| 인텐트가 빈 QA 수 | " + emptyIntent + " |");
        md.AppendLine("| 고유 질문 수 | " + uniqueQuestions + " |");
        md.AppendLine("| 동일 질문 중복 수(첫 행을 제외한 초과 행) | " + duplicateQuestionExcess + " |");
        md.AppendLine("| 질문·답변 완전 중복 수(첫 행을 제외한 초과 행) | " + exactDuplicateExcess + " |");
        md.AppendLine("| 질문은 같지만 답변이 다른 고유 질문 수 | " + differentAnswerGroups.Count + " |");
        md.AppendLine("| 위 조건에 해당하는 QA 행 수 | " + differentAnswerGroups.Sum(x => x.Count()) + " |\n");
        md.AppendLine("## 길이 통계\n");
        md.AppendLine("| 구분 | 최소 | 최대 | 평균 | 중앙값 |\n|---|---:|---:|---:|---:|");
        md.AppendLine("| 질문 | " + qMin + " | " + qMax + " | " + qAvg.ToString("0.00") + " | " + Median(train.Select(x => x.QuestionLength)).ToString("0.00") + " |");
        md.AppendLine("| 답변 | " + aMin + " | " + aMax + " | " + aAvg.ToString("0.00") + " | " + Median(train.Select(x => x.AnswerLength)).ToString("0.00") + " |\n");
        md.AppendLine("### 길이 구간별 개수\n\n| 구간 | 질문 | 답변 |\n|---|---:|---:|");
        foreach (string b in bucketOrder) md.AppendLine("| " + b + " | " + (qBuckets.ContainsKey(b) ? qBuckets[b] : 0) + " | " + (aBuckets.ContainsKey(b) ? aBuckets[b] : 0) + " |");
        md.AppendLine("\n## 인텐트별 QA 개수와 비율\n\n| 인텐트 | QA 수 | 비율 |\n|---|---:|---:|");
        foreach (var x in intentStats) md.AppendLine("| " + Md(x.Intent, 100) + " | " + x.Count + " | " + x.Ratio.ToString("0.0000") + "% |");
        md.AppendLine("\n## 질문 길이 4자 이하 실제 사례 50개\n\n| 질문 | 답변 | 인텐트 |\n|---|---|---|");
        foreach (Qa q in shortQuestions.Take(50)) md.AppendLine("| " + Md(q.Question, 120) + " | " + Md(q.Answer, 180) + " | " + Md(q.Intent, 80) + " |");
        md.AppendLine("\n## 답변 길이 4자 이하 실제 사례 50개\n\n| 질문 | 답변 | 인텐트 |\n|---|---|---|");
        foreach (Qa q in train.Where(x => x.AnswerLength <= 4).Take(50)) md.AppendLine("| " + Md(q.Question, 180) + " | " + Md(q.Answer, 120) + " | " + Md(q.Intent, 80) + " |");
        md.AppendLine("\n## 가장 자주 반복되는 질문 50개\n\n| 질문 | QA 수 | 서로 다른 답변 수 |\n|---|---:|---:|");
        foreach (var g in questionGroups.OrderByDescending(x => x.Count()).ThenBy(x => x.Key).Take(50)) md.AppendLine("| " + Md(g.Key, 180) + " | " + g.Count() + " | " + g.Select(x => x.Answer).Distinct().Count() + " |");
        md.AppendLine("\n## 질문은 같지만 답변이 다른 대표 사례 30개\n\n| 질문 | 서로 다른 답변 예시 |\n|---|---|");
        foreach (var g in differentAnswerGroups.OrderByDescending(x => x.Count()).Take(30)) md.AppendLine("| " + Md(g.Key, 160) + " | " + Md(string.Join(" / ", g.Select(x => x.Answer).Distinct().Take(3)), 300) + " |");
        md.AppendLine("\n## 지나치게 긴 질문과 답변 대표 사례\n\n길이가 긴 순서의 상위 사례이며, 자동 삭제 기준이 아닙니다.\n");
        md.AppendLine("### 긴 질문 상위 15개\n\n| 길이 | 질문 | 답변 |\n|---:|---|---|");
        foreach (Qa q in train.OrderByDescending(x => x.QuestionLength).Take(15)) md.AppendLine("| " + q.QuestionLength + " | " + Md(q.Question, 300) + " | " + Md(q.Answer, 180) + " |");
        md.AppendLine("\n### 긴 답변 상위 15개\n\n| 길이 | 질문 | 답변 |\n|---:|---|---|");
        foreach (Qa q in train.OrderByDescending(x => x.AnswerLength).Take(15)) md.AppendLine("| " + q.AnswerLength + " | " + Md(q.Question, 180) + " | " + Md(q.Answer, 300) + " |");
        md.AppendLine("\n## 제거 후보와 유지 후보\n");
        md.AppendLine("- **제거 후보:** 질문과 답변이 완전히 같은 중복 QA의 두 번째 이후 행, 의미가 불충분한 4자 이하 질문. 이번 분석에서는 삭제하지 않았습니다.");
        md.AppendLine("- **검토 후 유지 후보:** 질문이 같아도 답변이 다른 QA, 긴 질문·답변, 짧지만 문맥 없이도 의미가 분명한 질문입니다.");
        md.AppendLine("- 질문 중복은 상품·상황에 따라 답변이 달라질 수 있으므로 질문만 같다는 이유로 자동 삭제하면 안 됩니다.");
        File.WriteAllText(Path.Combine(docs, "super_data_quality.md"), md.ToString(), new UTF8Encoding(true));

        string design = "# 검색 실험 데이터셋 설계\n\n" +
            "## 왜 전체 130,516개를 바로 임베딩하지 않나?\n\n처음부터 전체를 처리하면 임베딩 시간과 저장 공간이 커지고, 전처리나 검색 설정의 작은 실수도 비싼 재작업으로 이어집니다. 먼저 약 5,000개로 검색 파이프라인과 평가 방법을 검증한 뒤 확대하는 편이 안전합니다.\n\n" +
            "## 왜 인텐트별 층화 표본추출을 사용하나?\n\n단순 무작위 추출은 데이터가 많은 인텐트에 치우칩니다. 이번 표본은 인텐트별 목록을 시드 42로 섞고 한 인텐트에서 한 건씩 순환 추출했으며, Training은 인텐트별 최대 100건, Validation은 최대 20건으로 제한했습니다. 희소 인텐트는 가능한 범위에서 모두 포함합니다. 실제 분포는 `results/*_sample_distribution.csv`에 기록했습니다.\n\n" +
            "## 왜 4자 이하 질문을 분리하나?\n\n`네`, `얼마요?`처럼 매우 짧은 질문은 단독 검색어로 의미가 부족한 경우가 많습니다. 기본 실험에서 제외하되 삭제하지 않고 별도 후보 파일에 보관하여 문맥 결합이나 별도 규칙 실험에 사용할 수 있게 했습니다.\n\n" +
            "## Training과 Validation의 역할\n\nTraining 표본은 나중에 Vector DB에 저장할 문서 후보입니다. Validation 표본은 저장하지 않고 검색 질의로만 사용하며, 원본 답변은 사람이 결과를 확인할 때 참고합니다. 이렇게 분리해야 평가 질문이 검색 문서에 그대로 들어가는 누수를 줄일 수 있습니다.\n\n" +
            "## Validation QA번호를 Training 정답 ID로 사용할 수 없는 이유\n\nQA번호와 상담번호는 각 데이터 분할 내부의 대화 묶음 식별자입니다. Validation의 번호가 Training의 동일한 의미 문서나 정답 문서를 가리킨다는 연결 정보가 없으므로 직접 매칭하면 잘못된 정답을 만들게 됩니다.\n\n" +
            "## 왜 첫 평가는 Intent Hit@k인가?\n\n현재 Validation 질문마다 정확히 어느 Training 문서가 정답인지 표시된 연결 라벨이 없습니다. 따라서 검색 상위 k개 안에 질문과 같은 인텐트의 문서가 하나라도 있는지를 보는 Intent Hit@k가 초기 검색 품질을 확인할 수 있는 현실적인 대리 지표입니다.\n\n" +
            "## 왜 이후 사람 지정 정답 평가셋이 필요한가?\n\n같은 인텐트라도 상품, 가격, 배송 조건이 다르면 실제로 유용한 답변은 달라집니다. Intent Hit@k는 주제 일치만 볼 뿐 내용의 정확성을 보장하지 않습니다. 최종적으로는 사람이 각 질문에 관련 Training 문서 ID를 하나 이상 지정한 평가셋을 만들어 Recall@k, MRR 같은 문서 단위 지표로 평가해야 합니다.\n\n" +
            "## 이번 실제 표본\n\n- Training 표본: " + trainSample.Count + "개\n- Training 짧은 질문 후보: " + shortCandidates.Count + "개\n- Validation 평가 표본: " + validationSample.Count + "개\n- 랜덤 시드: 42\n";
        File.WriteAllText(Path.Combine(docs, "experiment_dataset_design.md"), design, new UTF8Encoding(true));

        Console.WriteLine("Training 완전 QA: " + total);
        Console.WriteLine("빈 인텐트: " + emptyIntent);
        Console.WriteLine("고유 질문: " + uniqueQuestions + ", 동일 질문 초과 중복: " + duplicateQuestionExcess);
        Console.WriteLine("완전 중복 초과 행: " + exactDuplicateExcess + ", 같은 질문/다른 답변 질문: " + differentAnswerGroups.Count);
        Console.WriteLine("Training 표본: " + trainSample.Count + ", 짧은 질문 후보: " + shortCandidates.Count);
        Console.WriteLine("Validation 완전 QA 입력: " + validation.Count + ", 평가 표본: " + validationSample.Count);
    }
}
