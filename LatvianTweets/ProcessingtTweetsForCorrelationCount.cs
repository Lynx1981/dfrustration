using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace LatvianTweets
{
	class Program
	{
		static void Main(string[] args)
		{
			string[] lines0 = System.IO.File.ReadAllLines(@"C:\Users\Lynxa\Desktop\Doctorals\Latvian tweets\tweets_vl.txt");
			string[] lines1 = System.IO.File.ReadAllLines(@"C:\Users\Lynxa\Desktop\Doctorals\Latvian tweets\tweets_kl.txt");
			string[] lines2 = System.IO.File.ReadAllLines(@"C:\Users\Lynxa\Desktop\Doctorals\Latvian tweets\tweets_jb2.txt");
			string[] lines3 = System.IO.File.ReadAllLines(@"C:\Users\Lynxa\Desktop\Doctorals\Latvian tweets\tweets_ss2.txt");
			StreamWriter allAnnotations = new StreamWriter(@"C:\Users\Lynxa\Desktop\Doctorals\Latvian tweets\all_annotations_counts9.txt");
			StreamWriter uppercase = new StreamWriter(@"C:\Users\Lynxa\Desktop\Doctorals\Latvian tweets\uppercase.txt");

			string pattern = @"[A-ZĀĪĒŪŠĶĻŅČĢŽ]{5,}$";
			string patternA= @"[AaĀā]{3,}";
			string patternAllEmo = @"([<]).+?([>])";
			string patternHash = @"((\s|^)[#][\S]*?($|\s))";
			
			Regex multiLetterA = new Regex(patternA);
			Regex regexUpperCase = new Regex(pattern);
			Regex regexAllEmo = new Regex(patternAllEmo);
			Regex regexHash = new Regex(patternHash);
			
			Dictionary<string,int> result = new Dictionary<string,int>();

			for (int i = 0; i < lines0.Length; i++)
			{
				string line = lines0[i];
				string tline = "";
				if (i >= 1228)
				{
					;
				}
				if (line.Length > 0 && line[line.Length - 1].Equals('}'))
				{

					tline = line.Substring(0, line.LastIndexOf("{")) + "\t";
					if (line.LastIndexOf("}") - line.LastIndexOf("{") < 2)
						tline += "-";
					else
					{
						tline += line.Substring(line.LastIndexOf("{") + 1, 1);
					}
					tline += "\t";
					if (lines1[i].LastIndexOf("}") - lines1[i].LastIndexOf("{") < 2)
						tline += "-";
					else
					{
						tline += lines1[i].Substring(lines1[i].LastIndexOf("{") + 1, 1);
					}
					tline += "\t";
					if (lines2[i].LastIndexOf("}") - lines2[i].LastIndexOf("{") < 2)
						tline += "-";
					else
					{
						tline += lines2[i].Substring(lines2[i].LastIndexOf("{") + 1, 1);
					}
					tline += "\t";
					if (lines3[i].LastIndexOf("}") - lines3[i].LastIndexOf("{") < 2)
						tline += "-";
					else
					{
						tline += lines3[i].Substring(lines3[i].LastIndexOf("{") + 1, 1);
					}

					int countU = 0;
					
					List<string> list = lines0[i].Split(' ').ToList();

					foreach (string val in list)
					{
						if (val.Length > 20) continue;
						Match match = regexUpperCase.Match(val);
						if (match.Value.Length > 4 && val!="AM" && val!="PM")
						{
							if (result.ContainsKey(match.Value)) result[match.Value]++;
							else result.Add(match.Value, 1);
							countU++;						
						}
					}

					int countEmo = 0;
					int countEmoPos = 0;
					int countEmoNeu = 0;
					var mline = line;
					Match matchE= regexAllEmo.Match(mline);
					while (matchE.Value.Length > 3)
					{				
						countEmo++;
						mline = mline.Substring(mline.IndexOf(matchE.Value)+matchE.Value.Length);
						matchE = regexAllEmo.Match(mline);
						if (matchE.Value.ToUpper().Contains("SMIL") || matchE.Value.ToUpper().Contains("GRIN") || matchE.Value.ToUpper().Contains("LAUGH") 
							|| matchE.Value.ToUpper().Contains("TRIUMPH") || matchE.Value.ToUpper().Contains("JOY") || matchE.Value.Contains("Zany") 
							|| matchE.Value.Contains("savouring") || matchE.Value.Contains("Thumbs up") || matchE.Value.Contains("Latvia") || matchE.Value.ToUpper().Contains("HAPPY"))
						{
							countEmoPos++;
						}
					}

					mline = line;
					matchE = regexAllEmo.Match(mline);
					while (matchE.Value.Length > 3)
					{
						mline = mline.Substring(mline.IndexOf(matchE.Value) + matchE.Value.Length);
						matchE = regexAllEmo.Match(mline);
						if (matchE.Value.ToUpper().Contains("ARROW") || matchE.Value.ToUpper().Contains("FLAG") || matchE.Value.ToUpper().Contains("NERD")
							|| matchE.Value.ToUpper().Contains("MONKEY") || matchE.Value.ToUpper().Contains("FOLDED"))
						{
							countEmoNeu++;
						}
					}

					var countHEmoPos = (line.Contains(":)") || line.Contains(":-D") || line.Contains(";)")|| line.Contains(";-)") || line.Contains(":-P")) ? 1 : 0;
					var countHEmoNeg = (line.Contains(":(")) ? 1 : 0;
					var countEmoAllBool = countEmo> 0 ? 1 : 0;

					var countHash = 0;
					mline = line;
					Match matchH = regexHash.Match(mline);
					while (matchH.Value.Length > 2)
					{
						countHash++;
						mline = mline.Substring(mline.IndexOf(matchH.Value) + matchH.Value.Length);
						matchH = regexHash.Match(mline);
					}

					var countEmoPosBool = countEmoPos > 0 ? 1 : 0;
					var countHashBool = countHash > 0 ? 1 : 0;
					var countEmoNeg = countEmo - countEmoPos - countEmoNeu;
					var countEmoNegNeu = countEmo - countEmoPos;

					//number of exclamation signs					
					double countUN = (double)countU / (double)list.Count;

					var countExcl = line.Count(x => x == '!');
					double countExclN = (double) countExcl / (double) list.Count;

					var countQ = line.Count(x => x == '?');
					double countQN = (double) countQ / (double) list.Count;

					var countDot = line.Count(x => x == '.');
					double countDotN = (double) countDot / (double) list.Count;

					var countCom = line.Count(x => x == ',') > 0 ? line.Count(x => x == ',') - 1 : 0;
					double countComN = (double)countCom / (double)list.Count;

					var countQuo = line.Count(x => (x == '\"' || x=='\'' || x == '“' || x == '”' || x == '‘' || x == '’'));
					double countQuoN = (double)countQuo / (double)list.Count;
					
					Match matchA = multiLetterA.Match(line);
					var multiA = !line.Contains("https://pbs.twimg.com/")||line.Contains("https://pbs.twimg.com/media/EPsP9") ? (matchA.Length):0;

					var picture = line.Contains("https://pbs.twimg.com/") ? 1 : 0;
					var PTACgovLV = line.ToUpper().Contains("PTACGOVLV") ? 1 : 0;
					
					var length = line.Length;
	
					tline += String.Format("{0}{1}{0}{2:0.00}{0}{3}{0}{4:0.00}{0}{5}{0}{6:0.00}{0}{7}{0}{8:0.00}{0}{9}{0}{10:0.00}{0}{11}{0}{12:0.00}{0}{13}{0}{14}{0}{15}{0}{16:0.00}{0}{17}{0}{18}{0}" +
						"{19}{0}{20}{0}{21}{0}{22}{0}{23}{0}{24}{0}{25}{0}{26}",
						"\t", countU, countUN, countExcl, countExclN, countQ, countQN, countDot, countDotN, countCom, countComN, countQuo,countQuoN, multiA, countEmo, countEmoAllBool, countEmoPos, countEmoNeg, countEmoNeu, countEmoNegNeu, countHEmoPos, countHEmoNeg, countHash, countHashBool, picture, PTACgovLV,
						length);
					allAnnotations.WriteLine(tline);
				}
				else
				{
					//tline = line;
					tline = "";
				}				
			}

			foreach (var r in result)
			{
				uppercase.WriteLine(r.Key + "\t" + r.Value);
			}
			uppercase.Flush();

			allAnnotations.Flush();						
		}		
	}
}

