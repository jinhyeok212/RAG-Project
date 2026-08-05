using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

// PowerShell의 느린 동적 객체 반복을 피하기 위한 전용 CSV 처리기입니다.
public static class SuperQaProcessor
{
    private sealed class Item { public long Order; public string Text = ""; }
    private sealed class Group
    {
        public string Category = "", Conversation = "", QaNumber = "";
        public readonly List<Item> Questions = new List<Item>();
        public readonly List<Item> Answers = new List<Item>();
        public readonly List<Item> Intents = new List<Item>();
    }

    // 따옴표 안의 쉼표와 줄바꿈까지 인식하는 간단한 표준 CSV 파서입니다.
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
                row.Add(field.ToString()); field.Clear();
                yield return row.ToArray(); row.Clear();
            }
            else field.Append(ch);
        }
    }

    private static Tuple<Encoding, string> DetectEncoding(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
            return Tuple.Create<Encoding, string>(new UTF8Encoding(true, true), "UTF-8 with BOM");
        if (bytes.Length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE)
            return Tuple.Create<Encoding, string>(new UnicodeEncoding(false, true, true), "UTF-16 LE");
        if (bytes.Length >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF)
            return Tuple.Create<Encoding, string>(new UnicodeEncoding(true, true, true), "UTF-16 BE");
        try
        {
            var utf8 = new UTF8Encoding(false, true); utf8.GetString(bytes);
            return Tuple.Create<Encoding, string>(utf8, "UTF-8 (no BOM)");
        }
        catch (DecoderFallbackException)
        {
            return Tuple.Create(Encoding.GetEncoding(949), "CP949");
        }
    }

    private static string Csv(string value) { return "\"" + (value ?? "").Replace("\"", "\"\"") + "\""; }
    private static string JoinItems(List<Item> items)
    {
        items.Sort((x, y) => x.Order.CompareTo(y.Order));
        return string.Join(" ", items.Select(x => x.Text).Where(x => !string.IsNullOrWhiteSpace(x))).Trim();
    }

    public static void Run(string input, string output, string completeOutput, string report)
    {
        var detected = DetectEncoding(input);
        Console.WriteLine("감지된 인코딩: " + detected.Item2);
        var groups = new Dictionary<string, Group>(StringComparer.Ordinal);
        var conversations = new HashSet<string>(StringComparer.Ordinal);
        var samples = new List<string[]>();
        long utterances = 0;
        string[] headers;

        using (var reader = new StreamReader(input, detected.Item1, true))
        {
            using (var rows = ReadCsv(reader).GetEnumerator())
            {
                if (!rows.MoveNext()) throw new InvalidDataException("CSV가 비어 있습니다.");
                headers = rows.Current;
                var idx = headers.Select((name, i) => new { name, i }).ToDictionary(x => x.name, x => x.i);
                string[] required = { "발화자", "발화문", "카테고리", "QA번호", "QA여부", "인텐트", "상담번호", "상담내순번" };
                foreach (string name in required) if (!idx.ContainsKey(name)) throw new InvalidDataException("필수 컬럼이 없습니다: " + name);

                while (rows.MoveNext())
                {
                    string[] row = rows.Current; utterances++;
                    if (row.Length != headers.Length) throw new InvalidDataException(utterances + "번째 행의 컬럼 수가 다릅니다.");
                    if (samples.Count < 10) samples.Add(row);
                    string category = row[idx["카테고리"]].Trim(), conversation = row[idx["상담번호"]].Trim(), qa = row[idx["QA번호"]].Trim();
                    conversations.Add(conversation);
                    string key = category + "\u001f" + conversation + "\u001f" + qa;
                    Group group;
                    if (!groups.TryGetValue(key, out group))
                    {
                        group = new Group { Category = category, Conversation = conversation, QaNumber = qa };
                        groups.Add(key, group);
                    }
                    long order; long.TryParse(row[idx["상담내순번"]], out order);
                    var item = new Item { Order = order, Text = row[idx["발화문"]].Trim() };
                    string type = row[idx["QA여부"]].Trim().ToLowerInvariant();
                    if (type == "q") group.Questions.Add(item); else if (type == "a") group.Answers.Add(item);
                    string intent = row[idx["인텐트"]].Trim();
                    if (intent.Length > 0) group.Intents.Add(new Item { Order = order, Text = intent });
                }
            }
        }

        Console.WriteLine("\n실제 컬럼명 (" + headers.Length + "개):\n" + string.Join(", ", headers));
        Console.WriteLine("\n앞부분 10개 행:");
        int[] show = { Array.IndexOf(headers, "발화자"), Array.IndexOf(headers, "발화문"), Array.IndexOf(headers, "카테고리"), Array.IndexOf(headers, "QA번호"), Array.IndexOf(headers, "QA여부"), Array.IndexOf(headers, "인텐트"), Array.IndexOf(headers, "상담번호"), Array.IndexOf(headers, "상담내순번") };
        Console.WriteLine("발화자\t발화문\t카테고리\tQA번호\tQA여부\t인텐트\t상담번호\t상담내순번");
        foreach (var row in samples) Console.WriteLine(string.Join("\t", show.Select(i => row[i].Replace("\r", " ").Replace("\n", " "))));

        long both = 0, qOnly = 0, aOnly = 0, bothEmpty = 0, shortQ = 0, qSum = 0, aSum = 0;
        var intents = new Dictionary<string, long>(StringComparer.Ordinal);
        string csvHeader = "\"document_id\",\"category\",\"conversation_id\",\"qa_number\",\"intent\",\"question\",\"answer\",\"question_length\",\"answer_length\"";
        using (var all = new StreamWriter(output, false, new UTF8Encoding(true)))
        using (var complete = new StreamWriter(completeOutput, false, new UTF8Encoding(true)))
        {
            all.WriteLine(csvHeader); complete.WriteLine(csvHeader);
            foreach (Group g in groups.Values)
            {
                string question = JoinItems(g.Questions), answer = JoinItems(g.Answers);
                g.Intents.Sort((x, y) => x.Order.CompareTo(y.Order));
                string intent = g.Intents.Count == 0 ? "" : g.Intents[0].Text;
                int qLen = question.Length, aLen = answer.Length; qSum += qLen; aSum += aLen;
                if (qLen <= 4) shortQ++;
                string intentKey = intent.Length == 0 ? "(빈 인텐트)" : intent;
                intents[intentKey] = intents.ContainsKey(intentKey) ? intents[intentKey] + 1 : 1;
                bool hasQ = qLen > 0, hasA = aLen > 0;
                if (hasQ && hasA) both++; else if (hasQ) qOnly++; else if (hasA) aOnly++; else bothEmpty++;
                string documentId = Uri.EscapeDataString(g.Category) + "__" + Uri.EscapeDataString(g.Conversation) + "__" + Uri.EscapeDataString(g.QaNumber);
                string line = string.Join(",", new[] { Csv(documentId), Csv(g.Category), Csv(g.Conversation), Csv(g.QaNumber), Csv(intent), Csv(question), Csv(answer), Csv(qLen.ToString()), Csv(aLen.ToString()) });
                all.WriteLine(line); if (hasQ && hasA) complete.WriteLine(line);
            }
        }

        long groupCount = groups.Count, emptySide = qOnly + aOnly + bothEmpty;
        double avgQ = groupCount == 0 ? 0 : Math.Round((double)qSum / groupCount, 2), avgA = groupCount == 0 ? 0 : Math.Round((double)aSum / groupCount, 2);
        var intentLines = intents.OrderByDescending(x => x.Value).ThenBy(x => x.Key).Select(x => "- " + x.Key + ": " + x.Value).ToArray();
        string splitName = Path.GetFileName(input).IndexOf("validation", StringComparison.OrdinalIgnoreCase) >= 0 ? "Validation" : "Training";
        string markdown = "# 슈퍼 " + splitName + " 데이터 QA 전처리 분석\n\n" +
            "- 원본 파일: `" + input.Replace('\\', '/') + "`\n- 감지된 인코딩: **" + detected.Item2 + "**\n" +
            "- 그룹 기준: `카테고리 + 상담번호 + QA번호`\n- 그룹 내 정렬: `상담내순번` 오름차순\n" +
            "- 질문/답변 합치기: 각각 `QA여부=q` / `QA여부=a` 발화를 공백 하나로 연결\n" +
            "- 인텐트 선택: 그룹 내 첫 번째 비공백 인텐트\n- 평균 길이 모수: 빈 질문/답변을 포함한 전체 QA 그룹\n\n" +
            "## 전체 통계\n\n| 항목 | 개수 |\n|---|---:|\n" +
            "| 전체 발화 수 | " + utterances + " |\n| 전체 상담 수 | " + conversations.Count + " |\n| 전체 QA 그룹 수 | " + groupCount + " |\n" +
            "| 질문과 답변이 모두 존재하는 그룹 수 | " + both + " |\n| 질문만 있는 그룹 수 | " + qOnly + " |\n| 답변만 있는 그룹 수 | " + aOnly + " |\n" +
            "| 질문과 답변이 모두 빈 그룹 수 | " + bothEmpty + " |\n| 질문이나 답변이 비어 있는 그룹 수 | " + emptySide + " |\n" +
            "| 평균 질문 길이 | " + avgQ.ToString("0.00", CultureInfo.InvariantCulture) + "자 |\n| 평균 답변 길이 | " + avgA.ToString("0.00", CultureInfo.InvariantCulture) + "자 |\n" +
            "| 질문 길이가 4자 이하인 그룹 수 | " + shortQ + " |\n\n## 인텐트별 QA 개수\n\n" + string.Join("\n", intentLines) +
            "\n\n## 생성 파일\n\n- 전체 QA 그룹: `" + output.Replace('\\', '/') + "`\n- 질문과 답변이 모두 있는 QA: `" + completeOutput.Replace('\\', '/') + "`\n";
        File.WriteAllText(report, markdown, new UTF8Encoding(true));

        Console.WriteLine("\n=== 전처리 통계 ===");
        Console.WriteLine("전체 발화 수: " + utterances + "\n전체 상담 수: " + conversations.Count + "\n전체 QA 그룹 수: " + groupCount);
        Console.WriteLine("질문+답변 그룹 수: " + both + "\n질문만 있는 그룹 수: " + qOnly + "\n답변만 있는 그룹 수: " + aOnly);
        Console.WriteLine("질문과 답변이 모두 빈 그룹 수: " + bothEmpty + "\n질문이나 답변이 빈 그룹 수: " + emptySide);
        Console.WriteLine("평균 질문 길이: " + avgQ + "자\n평균 답변 길이: " + avgA + "자\n질문 길이 4자 이하: " + shortQ);
        Console.WriteLine("\n인텐트별 QA 개수:\n" + string.Join("\n", intentLines));
        Console.WriteLine("\n저장 완료: " + output + "\n완전한 QA 저장 완료: " + completeOutput + "\n분석 문서 저장 완료: " + report);
    }
}
