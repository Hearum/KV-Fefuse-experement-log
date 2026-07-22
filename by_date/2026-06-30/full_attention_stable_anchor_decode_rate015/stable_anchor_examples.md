# Stable Anchor Token Examples

- rate: 0.15
- frequency threshold: selected by >= 1.00 of queries
- source: MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability

## Example 0

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1170 | `'.\n'` | punct_or_symbol | 34/34 | 1.0 | through the fields and woods of the village northwards.\n |
| 2 | 1166 | `' the'` | latin_word | 34/34 | 2.7 | the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 3 | 1150 | `' the'` | latin_word | 34/34 | 3.4 | and are on the northern side of the plain of the River Kennet, with the Berkshire Downs rising through |
| 4 | 1147 | `' the'` | latin_word | 34/34 | 4.3 | homes are clustered and are on the northern side of the plain of the River Kennet, with the Berkshire |
| 5 | 1161 | `' the'` | latin_word | 34/34 | 5.0 | River Kennet, with the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 6 | 1169 | `'wards'` | latin_word | 34/34 | 5.7 | rising through the fields and woods of the village northwards.\n |
| 7 | 1154 | `','` | punct_or_symbol | 34/34 | 7.0 | northern side of the plain of the River Kennet, with the Berkshire Downs rising through the fields and woods |
| 8 | 1143 | `' the'` | latin_word | 34/34 | 9.3 | miles). The village homes are clustered and are on the northern side of the plain of the River Kennet |
| 9 | 1168 | `' north'` | latin_word | 34/34 | 9.9 | Downs rising through the fields and woods of the village northwards.\n |
| 10 | 1156 | `' the'` | latin_word | 34/34 | 10.9 | of the plain of the River Kennet, with the Berkshire Downs rising through the fields and woods of the |
| 11 | 1163 | `' and'` | latin_word | 34/34 | 11.2 | et, with the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 12 | 1134 | `').'` | punct_or_symbol | 34/34 | 11.5 | (8 miles) and Newbury (6 miles). The village homes are clustered and are on the northern |
| 13 | 1010 | `' a'` | latin_word | 34/34 | 12.7 | mouth within Norfolk\nDocument: The Rythe is a river or stream in north Surrey, England which is |
| 14 | 1158 | `' Downs'` | latin_word | 34/34 | 14.6 | plain of the River Kennet, with the Berkshire Downs rising through the fields and woods of the village north |
| 15 | 1167 | `' village'` | latin_word | 34/34 | 15.8 | Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 16 | 1157 | `' Berkshire'` | latin_word | 34/34 | 16.6 | the plain of the River Kennet, with the Berkshire Downs rising through the fields and woods of the village |
| 17 | 1104 | `'.'` | punct_or_symbol | 34/34 | 20.5 | a village and civil parish in West Berkshire, England. The village straddles the London to Bath ( |
| 18 | 1165 | `' of'` | latin_word | 34/34 | 19.1 | with the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 19 | 1110 | `' the'` | latin_word | 34/34 | 22.1 | West Berkshire, England. The village straddles the London to Bath (A4) road between the |
| 20 | 1164 | `' woods'` | latin_word | 34/34 | 21.4 | , with the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 21 | 1160 | `' through'` | latin_word | 34/34 | 21.9 | the River Kennet, with the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 22 | 1162 | `' fields'` | latin_word | 34/34 | 21.2 | Kennet, with the Berkshire Downs rising through the fields and woods of the village northwards.\n |
| 23 | 1155 | `' with'` | latin_word | 34/34 | 22.4 | side of the plain of the River Kennet, with the Berkshire Downs rising through the fields and woods of |
| 24 | 1120 | `' the'` | latin_word | 34/34 | 24.1 | the London to Bath (A4) road between the towns of Reading (8 miles) and Newbury |
| 25 | 1135 | `' The'` | latin_word | 34/34 | 24.8 | 8 miles) and Newbury (6 miles). The village homes are clustered and are on the northern side |
| 26 | 1140 | `' and'` | latin_word | 34/34 | 27.6 | bury (6 miles). The village homes are clustered and are on the northern side of the plain of the |
| 27 | 1159 | `' rising'` | latin_word | 34/34 | 27.7 | of the River Kennet, with the Berkshire Downs rising through the fields and woods of the village northwards |
| 28 | 881 | `' '` | space | 34/34 | 37.3 | 52.58798 ° N 1.64262 ° E ﻿ |
| 29 | 652 | `' reforms'` | latin_word | 34/34 | 37.5 | was part of National Road 13 but administrative reforms passed this part of the road to the county. |
| 30 | 1087 | `'.\n'` | punct_or_symbol | 34/34 | 31.4 | , a woodland of the mainly wooded Esher Commons.\nDocument: Woolhampton is a village and civil |
| 31 | 1117 | `')'` | punct_or_symbol | 34/34 | 34.0 | straddles the London to Bath (A4) road between the towns of Reading (8 miles) |
| 32 | 1102 | `','` | punct_or_symbol | 34/34 | 33.6 | pton is a village and civil parish in West Berkshire, England. The village straddles the London to |
| 33 | 1138 | `' are'` | latin_word | 34/34 | 35.8 | and Newbury (6 miles). The village homes are clustered and are on the northern side of the plain |
| 34 | 1151 | `' River'` | latin_word | 34/34 | 38.9 | are on the northern side of the plain of the River Kennet, with the Berkshire Downs rising through the |
| 35 | 1127 | `')'` | punct_or_symbol | 34/34 | 38.6 | ) road between the towns of Reading (8 miles) and Newbury (6 miles). The village homes |
| 36 | 1114 | `' ('` | punct_or_symbol | 34/34 | 38.6 | . The village straddles the London to Bath (A4) road between the towns of Reading ( |
| 37 | 1148 | `' plain'` | latin_word | 34/34 | 39.1 | are clustered and are on the northern side of the plain of the River Kennet, with the Berkshire Downs |
| 38 | 1062 | `' the'` | latin_word | 34/34 | 42.7 | ton and Long Ditton, then discharging into the Thames, its longest branch is the Arbrook which |
| 39 | 1152 | `' Kenn'` | latin_word | 34/34 | 41.5 | on the northern side of the plain of the River Kennet, with the Berkshire Downs rising through the fields |
| 40 | 1105 | `' The'` | latin_word | 34/34 | 43.1 | village and civil parish in West Berkshire, England. The village straddles the London to Bath (A |
| 41 | 1139 | `' clustered'` | latin_word | 34/34 | 43.3 | Newbury (6 miles). The village homes are clustered and are on the northern side of the plain of |
| 42 | 1144 | `' northern'` | latin_word | 34/34 | 43.7 | ). The village homes are clustered and are on the northern side of the plain of the River Kennet, |
| 43 | 1149 | `' of'` | latin_word | 34/34 | 44.5 | clustered and are on the northern side of the plain of the River Kennet, with the Berkshire Downs rising |
| 44 | 1094 | `' a'` | latin_word | 34/34 | 45.9 | Esher Commons.\nDocument: Woolhampton is a village and civil parish in West Berkshire, England. |
| 45 | 1137 | `' homes'` | latin_word | 34/34 | 45.4 | ) and Newbury (6 miles). The village homes are clustered and are on the northern side of the |
| 46 | 1029 | `' feature'` | latin_word | 34/34 | 51.6 | which is generally open and which is a natural woodland feature for approximately half of its course before being variously |
| 47 | 1141 | `' are'` | latin_word | 34/34 | 49.8 | (6 miles). The village homes are clustered and are on the northern side of the plain of the River |
| 48 | 1145 | `' side'` | latin_word | 34/34 | 49.9 | The village homes are clustered and are on the northern side of the plain of the River Kennet, with |
| 49 | 1142 | `' on'` | latin_word | 34/34 | 51.3 | 6 miles). The village homes are clustered and are on the northern side of the plain of the River Kenn |
| 50 | 1128 | `' and'` | latin_word | 34/34 | 53.4 | road between the towns of Reading (8 miles) and Newbury (6 miles). The village homes are |
| 51 | 1153 | `'et'` | latin_word | 34/34 | 54.6 | the northern side of the plain of the River Kennet, with the Berkshire Downs rising through the fields and |
| 52 | 1136 | `' village'` | latin_word | 34/34 | 55.6 | miles) and Newbury (6 miles). The village homes are clustered and are on the northern side of |
| 53 | 1112 | `' to'` | latin_word | 34/34 | 62.6 | , England. The village straddles the London to Bath (A4) road between the towns of |
| 54 | 752 | `' '` | space | 34/34 | 61.1 | 9 m (259 ft) - coordinates 52 ° 37 ′ 1 |
| 55 | 1146 | `' of'` | latin_word | 34/34 | 57.4 | village homes are clustered and are on the northern side of the plain of the River Kennet, with the |
| 56 | 1093 | `' is'` | latin_word | 34/34 | 61.1 | wooded Esher Commons.\nDocument: Woolhampton is a village and civil parish in West Berkshire, England |
| 57 | 1124 | `' ('` | punct_or_symbol | 34/34 | 58.9 | (A4) road between the towns of Reading (8 miles) and Newbury (6 miles). |
| 58 | 1131 | `' ('` | punct_or_symbol | 34/34 | 61.1 | towns of Reading (8 miles) and Newbury (6 miles). The village homes are clustered and are |
| 59 | 1003 | `'\n'` | newline_or_space | 34/34 | 62.2 | 2 mi) Location of the river mouth within Norfolk\nDocument: The Rythe is a river or stream |
| 60 | 1089 | `':'` | punct_or_symbol | 34/34 | 62.6 | woodland of the mainly wooded Esher Commons.\nDocument: Woolhampton is a village and civil parish in |
| 61 | 1081 | `' the'` | latin_word | 34/34 | 64.7 | brook which drains Arbrook Common, a woodland of the mainly wooded Esher Commons.\nDocument: Woolham |
| 62 | 1103 | `' England'` | latin_word | 34/34 | 68.7 | is a village and civil parish in West Berkshire, England. The village straddles the London to Bath |
| 63 | 1115 | `'A'` | latin_word | 34/34 | 69.5 | The village straddles the London to Bath (A4) road between the towns of Reading (8 |
| 64 | 1047 | `','` | punct_or_symbol | 34/34 | 71.0 | being variously culverted and a suburban garden feature, passing between Thames Ditton and Long Ditton, |
| 65 | 1121 | `' towns'` | latin_word | 34/34 | 71.3 | London to Bath (A4) road between the towns of Reading (8 miles) and Newbury ( |
| 66 | 1077 | `','` | punct_or_symbol | 34/34 | 72.2 | branch is the Arbrook which drains Arbrook Common, a woodland of the mainly wooded Esher Commons.\n |
| 67 | 1096 | `' and'` | latin_word | 34/34 | 73.7 | Commons.\nDocument: Woolhampton is a village and civil parish in West Berkshire, England. The village |
| 68 | 1132 | `'6'` | number | 34/34 | 75.8 | of Reading (8 miles) and Newbury (6 miles). The village homes are clustered and are on |
| 69 | 1109 | `'les'` | latin_word | 34/34 | 77.3 | in West Berkshire, England. The village straddles the London to Bath (A4) road between |
| 70 | 1118 | `' road'` | latin_word | 34/34 | 76.2 | addles the London to Bath (A4) road between the towns of Reading (8 miles) and |
| 71 | 1111 | `' London'` | latin_word | 34/34 | 79.8 | Berkshire, England. The village straddles the London to Bath (A4) road between the towns |
| 72 | 1101 | `' Berkshire'` | latin_word | 34/34 | 78.3 | hampton is a village and civil parish in West Berkshire, England. The village straddles the London |
| 73 | 1035 | `' course'` | latin_word | 34/34 | 79.4 | is a natural woodland feature for approximately half of its course before being variously culverted and a suburban garden |
| 74 | 1098 | `' parish'` | latin_word | 34/34 | 81.0 | Document: Woolhampton is a village and civil parish in West Berkshire, England. The village stradd |
| 75 | 1042 | `' and'` | latin_word | 34/34 | 84.4 | half of its course before being variously culverted and a suburban garden feature, passing between Thames Ditton |
| 76 | 1123 | `' Reading'` | latin_word | 34/34 | 82.0 | Bath (A4) road between the towns of Reading (8 miles) and Newbury (6 miles |
| 77 | 931 | `'3'` | number | 34/34 | 86.3 | � 17 ''N 1 ° 38 ′ 33'' E ﻿ / |
| 78 | 1122 | `' of'` | latin_word | 34/34 | 83.3 | to Bath (A4) road between the towns of Reading (8 miles) and Newbury (6 |
| 79 | 1130 | `'bury'` | latin_word | 34/34 | 83.1 | the towns of Reading (8 miles) and Newbury (6 miles). The village homes are clustered and |
| 80 | 1119 | `' between'` | latin_word | 34/34 | 85.9 | les the London to Bath (A4) road between the towns of Reading (8 miles) and New |

## Example 1

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1204 | `'.\n'` | punct_or_symbol | 34/34 | 1.0 | are located in Vista, California, Switzerland and Japan.\n |
| 2 | 1097 | `' the'` | latin_word | 34/34 | 2.2 | cakes, the most famous being a realistic replica of the Sydney Opera House for Australia Day 201 |
| 3 | 834 | `' '` | space | 34/34 | 4.0 | linville, Illinois. The building was constructed between 1909 and 1910 |
| 4 | 1198 | `','` | punct_or_symbol | 34/34 | 3.8 | . NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 5 | 1202 | `' and'` | latin_word | 34/34 | 4.4 | manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 6 | 1076 | `' They'` | latin_word | 34/34 | 6.2 | , as well as for television and many magazines. They have also created a host of ‘stunt’ |
| 7 | 1200 | `','` | punct_or_symbol | 34/34 | 7.1 | AI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 8 | 1188 | `'.'` | punct_or_symbol | 34/34 | 8.0 | 241 employees in 2007. NAI's manufacturing facilities are located in Vista, |
| 9 | 1203 | `' Japan'` | latin_word | 34/34 | 8.4 | facilities are located in Vista, California, Switzerland and Japan.\n |
| 10 | 1194 | `' are'` | latin_word | 34/34 | 10.2 | 2007. NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 11 | 1201 | `' Switzerland'` | latin_word | 34/34 | 11.7 | 's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 12 | 1196 | `' in'` | latin_word | 34/34 | 11.9 | 07. NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 13 | 1195 | `' located'` | latin_word | 34/34 | 14.5 | 007. NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 14 | 1156 | `'.'` | punct_or_symbol | 34/34 | 15.3 | , California which manufactures nutritional supplements such as Juice Plus. NAI was founded in 1980 |
| 15 | 1199 | `' California'` | latin_word | 34/34 | 15.5 | NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 16 | 1174 | `','` | punct_or_symbol | 34/34 | 16.4 | 980 by Mark A. LeDoux, and had 241 employees in 2 |
| 17 | 1087 | `' cakes'` | latin_word | 34/34 | 19.3 | have also created a host of ‘stunt’ cakes, the most famous being a realistic replica of the |
| 18 | 1193 | `' facilities'` | latin_word | 34/34 | 17.9 | 2007. NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan |
| 19 | 1191 | `"'s"` | other_text | 34/34 | 18.7 | employees in 2007. NAI's manufacturing facilities are located in Vista, California, Switzerland |
| 20 | 1197 | `' Vista'` | latin_word | 34/34 | 19.7 | 7. NAI's manufacturing facilities are located in Vista, California, Switzerland and Japan.\n |
| 21 | 1192 | `' manufacturing'` | latin_word | 34/34 | 21.6 | in 2007. NAI's manufacturing facilities are located in Vista, California, Switzerland and |
| 22 | 1183 | `' '` | space | 34/34 | 23.5 | oux, and had 241 employees in 2007. NAI's manufacturing facilities |
| 23 | 1134 | `' ('` | punct_or_symbol | 34/34 | 26.1 | decorators to make.\nDocument: Natural Alternatives International (NAI) is an American company based in San |
| 24 | 1146 | `','` | punct_or_symbol | 34/34 | 26.3 | I) is an American company based in San Marcos, California which manufactures nutritional supplements such as Juice Plus. |
| 25 | 1181 | `' employees'` | latin_word | 34/34 | 26.9 | LeDoux, and had 241 employees in 2007. NAI's |
| 26 | 1139 | `' an'` | latin_word | 34/34 | 28.9 | : Natural Alternatives International (NAI) is an American company based in San Marcos, California which manufactures |
| 27 | 1189 | `' N'` | latin_word | 34/34 | 27.7 | 41 employees in 2007. NAI's manufacturing facilities are located in Vista, California |
| 28 | 1177 | `' '` | space | 34/34 | 28.8 | by Mark A. LeDoux, and had 241 employees in 2007 |
| 29 | 1175 | `' and'` | latin_word | 34/34 | 30.4 | 80 by Mark A. LeDoux, and had 241 employees in 20 |
| 30 | 1022 | `' '` | space | 34/34 | 36.5 | private home.\nDocument: Planet Cake has created over 12000 couture cakes, including |
| 31 | 1127 | `'.\n'` | punct_or_symbol | 34/34 | 32.2 | tons and required 32 cake decorators to make.\nDocument: Natural Alternatives International (NAI) |
| 32 | 1176 | `' had'` | latin_word | 34/34 | 34.3 | 0 by Mark A. LeDoux, and had 241 employees in 200 |
| 33 | 1190 | `'AI'` | latin_word | 34/34 | 35.7 | 1 employees in 2007. NAI's manufacturing facilities are located in Vista, California, |
| 34 | 1182 | `' in'` | latin_word | 34/34 | 36.6 | Doux, and had 241 employees in 2007. NAI's manufacturing |
| 35 | 1078 | `' also'` | latin_word | 34/34 | 40.8 | well as for television and many magazines. They have also created a host of ‘stunt’ cakes, |
| 36 | 846 | `' a'` | latin_word | 34/34 | 40.0 | 909 and 1910 as a meetinghouse for Carlinville's chapter of the |
| 37 | 1184 | `'2'` | number | 34/34 | 38.5 | , and had 241 employees in 2007. NAI's manufacturing facilities are |
| 38 | 1137 | `')'` | punct_or_symbol | 34/34 | 40.0 | .\nDocument: Natural Alternatives International (NAI) is an American company based in San Marcos, California |
| 39 | 1170 | `'.'` | punct_or_symbol | 34/34 | 39.6 | founded in 1980 by Mark A. LeDoux, and had 241 |
| 40 | 1162 | `' '` | space | 34/34 | 40.1 | such as Juice Plus. NAI was founded in 1980 by Mark A. LeD |
| 41 | 1179 | `'4'` | number | 34/34 | 41.2 | A. LeDoux, and had 241 employees in 2007. N |
| 42 | 1187 | `'7'` | number | 34/34 | 42.4 | 241 employees in 2007. NAI's manufacturing facilities are located in Vista |
| 43 | 132 | `'9'` | number | 34/34 | 51.4 | and sorbet. It was founded in 1978 in Burlington, Vermont, and operates globally |
| 44 | 1093 | `' a'` | latin_word | 34/34 | 47.8 | ‘stunt’ cakes, the most famous being a realistic replica of the Sydney Opera House for Australia Day |
| 45 | 1167 | `' by'` | latin_word | 34/34 | 46.5 | NAI was founded in 1980 by Mark A. LeDoux, and had |
| 46 | 1159 | `' was'` | latin_word | 34/34 | 46.6 | manufactures nutritional supplements such as Juice Plus. NAI was founded in 1980 by Mark A |
| 47 | 1178 | `'2'` | number | 34/34 | 47.4 | Mark A. LeDoux, and had 241 employees in 2007. |
| 48 | 1138 | `' is'` | latin_word | 34/34 | 50.2 | Document: Natural Alternatives International (NAI) is an American company based in San Marcos, California which |
| 49 | 1160 | `' founded'` | latin_word | 34/34 | 51.4 | nutritional supplements such as Juice Plus. NAI was founded in 1980 by Mark A. |
| 50 | 1157 | `' N'` | latin_word | 34/34 | 53.3 | California which manufactures nutritional supplements such as Juice Plus. NAI was founded in 1980 by |
| 51 | 1185 | `'0'` | number | 34/34 | 52.3 | and had 241 employees in 2007. NAI's manufacturing facilities are located |
| 52 | 836 | `'9'` | number | 34/34 | 56.6 | , Illinois. The building was constructed between 1909 and 1910 as a |
| 53 | 933 | `' H'` | latin_word | 34/34 | 55.5 | . Louis architectural firm of Helfensteller, Hirsch & Watson to design five classes of buildings which |
| 54 | 1186 | `'0'` | number | 34/34 | 53.2 | had 241 employees in 2007. NAI's manufacturing facilities are located in |
| 55 | 597 | `' a'` | latin_word | 34/34 | 61.3 | 19th century, reports described the area as a promising wilderness. The area was also characterized as " |
| 56 | 1173 | `'oux'` | latin_word | 34/34 | 57.1 | 1980 by Mark A. LeDoux, and had 241 employees in |
| 57 | 1163 | `'1'` | number | 34/34 | 57.4 | as Juice Plus. NAI was founded in 1980 by Mark A. LeDoux |
| 58 | 1129 | `':'` | punct_or_symbol | 34/34 | 57.8 | required 32 cake decorators to make.\nDocument: Natural Alternatives International (NAI) is an |
| 59 | 1180 | `'1'` | number | 34/34 | 59.0 | . LeDoux, and had 241 employees in 2007. NAI |
| 60 | 953 | `' Car'` | latin_word | 34/34 | 69.5 | which the League would use as meetinghouses. The Carlinville Chapter House is an example of a Class |
| 61 | 1075 | `'.'` | punct_or_symbol | 34/34 | 59.5 | few, as well as for television and many magazines. They have also created a host of ‘stunt |
| 62 | 1105 | `'2'` | number | 34/34 | 68.6 | replica of the Sydney Opera House for Australia Day 2011, which weighed over 1. |
| 63 | 1161 | `' in'` | latin_word | 34/34 | 64.4 | supplements such as Juice Plus. NAI was founded in 1980 by Mark A. Le |
| 64 | 1168 | `' Mark'` | latin_word | 34/34 | 63.8 | AI was founded in 1980 by Mark A. LeDoux, and had 2 |
| 65 | 1149 | `' manufactures'` | latin_word | 34/34 | 64.7 | an American company based in San Marcos, California which manufactures nutritional supplements such as Juice Plus. NAI was |
| 66 | 1169 | `' A'` | latin_word | 34/34 | 69.6 | was founded in 1980 by Mark A. LeDoux, and had 24 |
| 67 | 1151 | `' supplements'` | latin_word | 34/34 | 67.7 | company based in San Marcos, California which manufactures nutritional supplements such as Juice Plus. NAI was founded in |
| 68 | 1148 | `' which'` | latin_word | 34/34 | 68.6 | is an American company based in San Marcos, California which manufactures nutritional supplements such as Juice Plus. NAI |
| 69 | 1143 | `' in'` | latin_word | 34/34 | 69.6 | International (NAI) is an American company based in San Marcos, California which manufactures nutritional supplements such as |
| 70 | 1171 | `' Le'` | latin_word | 34/34 | 70.0 | in 1980 by Mark A. LeDoux, and had 241 employees |
| 71 | 1150 | `' nutritional'` | latin_word | 34/34 | 72.3 | American company based in San Marcos, California which manufactures nutritional supplements such as Juice Plus. NAI was founded |
| 72 | 1141 | `' company'` | latin_word | 34/34 | 73.1 | Alternatives International (NAI) is an American company based in San Marcos, California which manufactures nutritional supplements |
| 73 | 1172 | `'D'` | latin_word | 34/34 | 75.7 | 1980 by Mark A. LeDoux, and had 241 employees in |
| 74 | 1088 | `','` | punct_or_symbol | 34/34 | 79.6 | also created a host of ‘stunt’ cakes, the most famous being a realistic replica of the Sydney |
| 75 | 1166 | `'0'` | number | 34/34 | 77.0 | . NAI was founded in 1980 by Mark A. LeDoux, and had |
| 76 | 1080 | `' a'` | latin_word | 34/34 | 87.2 | for television and many magazines. They have also created a host of ‘stunt’ cakes, the most |
| 77 | 1153 | `' as'` | latin_word | 34/34 | 82.7 | in San Marcos, California which manufactures nutritional supplements such as Juice Plus. NAI was founded in 1 |
| 78 | 1142 | `' based'` | latin_word | 34/34 | 82.3 | atives International (NAI) is an American company based in San Marcos, California which manufactures nutritional supplements such |
| 79 | 1109 | `','` | punct_or_symbol | 34/34 | 82.4 | Opera House for Australia Day 2011, which weighed over 1.3 tons and required |
| 80 | 1154 | `' Juice'` | latin_word | 34/34 | 82.9 | San Marcos, California which manufactures nutritional supplements such as Juice Plus. NAI was founded in 19 |

## Example 2

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1376 | `').\n'` | punct_or_symbol | 34/34 | 1.0 | 50 square miles (130 km2).\n |
| 2 | 1359 | `' the'` | latin_word | 34/34 | 2.2 | habited training grounds. The actual inhabited area for the city is slightly more than 50 square miles |
| 3 | 1371 | `'1'` | number | 34/34 | 4.1 | is slightly more than 50 square miles (130 km2).\n |
| 4 | 1353 | `'.'` | punct_or_symbol | 34/34 | 4.3 | much of which consists of uninhabited training grounds. The actual inhabited area for the city is slightly more |
| 5 | 1291 | `'.'` | punct_or_symbol | 34/34 | 7.3 | .4 km2) is land and 2.7 square miles (7.0 km2) |
| 6 | 1370 | `' ('` | punct_or_symbol | 34/34 | 5.8 | city is slightly more than 50 square miles (130 km2).\n |
| 7 | 1354 | `' The'` | latin_word | 34/34 | 7.7 | of which consists of uninhabited training grounds. The actual inhabited area for the city is slightly more than |
| 8 | 1375 | `'2'` | number | 34/34 | 9.3 | 50 square miles (130 km2).\n |
| 9 | 1365 | `' '` | space | 34/34 | 9.3 | actual inhabited area for the city is slightly more than 50 square miles (130 km2 |
| 10 | 1347 | `' of'` | latin_word | 34/34 | 12.8 | the Fort Jackson Military Installation, much of which consists of uninhabited training grounds. The actual inhabited area |
| 11 | 1361 | `' is'` | latin_word | 34/34 | 13.1 | training grounds. The actual inhabited area for the city is slightly more than 50 square miles (1 |
| 12 | 1372 | `'3'` | number | 34/34 | 13.8 | slightly more than 50 square miles (130 km2).\n |
| 13 | 1374 | `' km'` | latin_word | 34/34 | 14.1 | than 50 square miles (130 km2).\n |
| 14 | 1337 | `' the'` | latin_word | 34/34 | 15.2 | (210 km2), is contained within the Fort Jackson Military Installation, much of which consists of |
| 15 | 1328 | `'2'` | number | 34/34 | 20.3 | area, 81.2 square miles (210 km2), is contained within the Fort |
| 16 | 881 | `','` | punct_or_symbol | 34/34 | 18.6 | uru, Karnataka, India Number of locations 1,556 (17 October 20 |
| 17 | 1373 | `'0'` | number | 34/34 | 18.3 | more than 50 square miles (130 km2).\n |
| 18 | 1342 | `','` | punct_or_symbol | 34/34 | 18.3 | 2), is contained within the Fort Jackson Military Installation, much of which consists of uninhabited training grounds |
| 19 | 1358 | `' for'` | latin_word | 34/34 | 21.8 | uninhabited training grounds. The actual inhabited area for the city is slightly more than 50 square |
| 20 | 1366 | `'5'` | number | 34/34 | 21.9 | inhabited area for the city is slightly more than 50 square miles (130 km2).\n |
| 21 | 1357 | `' area'` | latin_word | 34/34 | 23.6 | of uninhabited training grounds. The actual inhabited area for the city is slightly more than 50 |
| 22 | 1369 | `' miles'` | latin_word | 34/34 | 24.2 | the city is slightly more than 50 square miles (130 km2).\n |
| 23 | 1356 | `' inhabited'` | latin_word | 34/34 | 25.5 | consists of uninhabited training grounds. The actual inhabited area for the city is slightly more than 5 |
| 24 | 1360 | `' city'` | latin_word | 34/34 | 26.5 | ited training grounds. The actual inhabited area for the city is slightly more than 50 square miles ( |
| 25 | 1364 | `' than'` | latin_word | 34/34 | 26.4 | The actual inhabited area for the city is slightly more than 50 square miles (130 km |
| 26 | 1311 | `' '` | space | 34/34 | 27.3 | ) is water (2.01%). Approximately ⅔ of Columbia's land area, 8 |
| 27 | 1363 | `' more'` | latin_word | 34/34 | 27.9 | . The actual inhabited area for the city is slightly more than 50 square miles (130 |
| 28 | 1319 | `','` | punct_or_symbol | 34/34 | 30.5 | %). Approximately ⅔ of Columbia's land area, 81.2 square miles (21 |
| 29 | 1355 | `' actual'` | latin_word | 34/34 | 30.8 | which consists of uninhabited training grounds. The actual inhabited area for the city is slightly more than |
| 30 | 1368 | `' square'` | latin_word | 34/34 | 31.9 | for the city is slightly more than 50 square miles (130 km2).\n |
| 31 | 832 | `'0'` | number | 34/34 | 40.4 | company) ISIN INE335K01011 Industry Coffeehouse Founded 1 |
| 32 | 1362 | `' slightly'` | latin_word | 34/34 | 33.0 | grounds. The actual inhabited area for the city is slightly more than 50 square miles (13 |
| 33 | 1259 | `'3'` | number | 34/34 | 39.7 | of 134.9 square miles (349.5 km2), of which |
| 34 | 1306 | `'.'` | punct_or_symbol | 34/34 | 39.5 | 7.0 km2) is water (2.01%). Approximately ⅔ of Columbia's |
| 35 | 1333 | `'),'` | punct_or_symbol | 34/34 | 35.3 | .2 square miles (210 km2), is contained within the Fort Jackson Military Installation, much |
| 36 | 1367 | `'0'` | number | 34/34 | 38.3 | area for the city is slightly more than 50 square miles (130 km2).\n |
| 37 | 1352 | `' grounds'` | latin_word | 34/34 | 38.3 | , much of which consists of uninhabited training grounds. The actual inhabited area for the city is slightly |
| 38 | 1233 | `' order'` | latin_word | 34/34 | 42.5 | boro series). All belong to the Ultisol soil order.According to the United States Census Bureau, the |
| 39 | 1351 | `' training'` | latin_word | 34/34 | 42.1 | Installation, much of which consists of uninhabited training grounds. The actual inhabited area for the city is |
| 40 | 1350 | `'ited'` | latin_word | 34/34 | 45.5 | Military Installation, much of which consists of uninhabited training grounds. The actual inhabited area for the city |
| 41 | 1327 | `' ('` | punct_or_symbol | 34/34 | 44.3 | land area, 81.2 square miles (210 km2), is contained within the |
| 42 | 1346 | `' consists'` | latin_word | 34/34 | 46.2 | within the Fort Jackson Military Installation, much of which consists of uninhabited training grounds. The actual inhabited |
| 43 | 1309 | `'%).'` | punct_or_symbol | 34/34 | 47.4 | km2) is water (2.01%). Approximately ⅔ of Columbia's land area, |
| 44 | 1348 | `' unin'` | latin_word | 34/34 | 48.6 | Fort Jackson Military Installation, much of which consists of uninhabited training grounds. The actual inhabited area for |
| 45 | 1043 | `','` | punct_or_symbol | 34/34 | 52.2 | upland Piedmont region and the Atlantic Coastal Plain, across which rivers drop as falls or rapids. |
| 46 | 1258 | `' ('` | punct_or_symbol | 34/34 | 74.7 | area of 134.9 square miles (349.5 km2), of which |
| 47 | 1344 | `' of'` | latin_word | 34/34 | 50.1 | is contained within the Fort Jackson Military Installation, much of which consists of uninhabited training grounds. The |
| 48 | 1339 | `' Jackson'` | latin_word | 34/34 | 51.1 | 10 km2), is contained within the Fort Jackson Military Installation, much of which consists of uninhab |
| 49 | 1345 | `' which'` | latin_word | 34/34 | 53.3 | contained within the Fort Jackson Military Installation, much of which consists of uninhabited training grounds. The actual |
| 50 | 1305 | `'2'` | number | 34/34 | 63.1 | (7.0 km2) is water (2.01%). Approximately ⅔ of Columbia |
| 51 | 1336 | `' within'` | latin_word | 34/34 | 54.5 | miles (210 km2), is contained within the Fort Jackson Military Installation, much of which consists |
| 52 | 1243 | `' the'` | latin_word | 34/34 | 58.0 | order.According to the United States Census Bureau, the city has a total area of 134 |
| 53 | 1320 | `' '` | space | 34/34 | 58.2 | Approximately ⅔ of Columbia's land area, 81.2 square miles (210 |
| 54 | 1340 | `' Military'` | latin_word | 34/34 | 57.1 | 0 km2), is contained within the Fort Jackson Military Installation, much of which consists of uninhabited |
| 55 | 1341 | `' Installation'` | latin_word | 34/34 | 57.9 | km2), is contained within the Fort Jackson Military Installation, much of which consists of uninhabited training |
| 56 | 1334 | `' is'` | latin_word | 34/34 | 60.4 | 2 square miles (210 km2), is contained within the Fort Jackson Military Installation, much of |
| 57 | 1323 | `'.'` | punct_or_symbol | 34/34 | 60.2 | � of Columbia's land area, 81.2 square miles (210 km2), |
| 58 | 1338 | `' Fort'` | latin_word | 34/34 | 62.0 | 210 km2), is contained within the Fort Jackson Military Installation, much of which consists of unin |
| 59 | 1316 | `"'s"` | other_text | 34/34 | 64.4 | .01%). Approximately ⅔ of Columbia's land area, 81.2 square miles |
| 60 | 1278 | `'3'` | number | 34/34 | 71.5 | which 132.2 square miles (342.4 km2) is land and |
| 61 | 1349 | `'hab'` | latin_word | 34/34 | 66.5 | Jackson Military Installation, much of which consists of uninhabited training grounds. The actual inhabited area for the |
| 62 | 584 | `' a'` | latin_word | 34/34 | 91.9 | (Green Creek tributary): Mud Run is a tributary of Green Creek in Columbia County, |
| 63 | 1242 | `','` | punct_or_symbol | 34/34 | 68.8 | soil order.According to the United States Census Bureau, the city has a total area of 13 |
| 64 | 1343 | `' much'` | latin_word | 34/34 | 69.1 | ), is contained within the Fort Jackson Military Installation, much of which consists of uninhabited training grounds. |
| 65 | 1237 | `' the'` | latin_word | 34/34 | 68.9 | belong to the Ultisol soil order.According to the United States Census Bureau, the city has a total |
| 66 | 1313 | `'�'` | punct_or_symbol | 34/34 | 71.4 | water (2.01%). Approximately ⅔ of Columbia's land area, 81. |
| 67 | 1335 | `' contained'` | latin_word | 34/34 | 73.0 | square miles (210 km2), is contained within the Fort Jackson Military Installation, much of which |
| 68 | 1317 | `' land'` | latin_word | 34/34 | 76.7 | 01%). Approximately ⅔ of Columbia's land area, 81.2 square miles ( |
| 69 | 729 | `' the'` | latin_word | 34/34 | 90.8 | is the westernmost point marked on most maps of the elongated archipelago that makes up this park |
| 70 | 1229 | `' the'` | latin_word | 34/34 | 78.4 | clay (Marlboro series). All belong to the Ultisol soil order.According to the United States |
| 71 | 1315 | `' Columbia'` | latin_word | 34/34 | 79.2 | 2.01%). Approximately ⅔ of Columbia's land area, 81.2 square |
| 72 | 1190 | `' be'` | latin_word | 34/34 | 84.9 | sand topsoil. The subsoil may be yellowish red sandy clay loam (Orangeburg |
| 73 | 1312 | `'�'` | punct_or_symbol | 34/34 | 80.8 | is water (2.01%). Approximately ⅔ of Columbia's land area, 81 |
| 74 | 806 | `' '` | space | 34/34 | 85.0 | Coffee Day Enterprises Limited Traded as BSE: 539436 NSE: CO |
| 75 | 1295 | `' ('` | punct_or_symbol | 34/34 | 98.0 | ) is land and 2.7 square miles (7.0 km2) is water (2 |
| 76 | 862 | `','` | punct_or_symbol | 34/34 | 97.6 | (1996) Headquarters Coffee Day Square, Vittal Mallya Road, Bengaluru, |
| 77 | 848 | `'2'` | number | 34/34 | 95.0 | Coffeehouse Founded 1996; 22 years ago (1996) Headquarters |
| 78 | 1058 | `' the'` | latin_word | 34/34 | 86.3 | as falls or rapids. Columbia grew up at the fall line of the Congaree River, which |
| 79 | 1318 | `' area'` | latin_word | 34/34 | 83.9 | 1%). Approximately ⅔ of Columbia's land area, 81.2 square miles (2 |
| 80 | 1304 | `' ('` | punct_or_symbol | 34/34 | 98.1 | miles (7.0 km2) is water (2.01%). Approximately ⅔ of |

## Example 3

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1242 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | Castle on several occasions during the absence of her spouse.\n |
| 2 | 1237 | `' the'` | latin_word | 35/35 | 2.3 | , she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 3 | 1235 | `' occasions'` | latin_word | 35/35 | 2.9 | 34, she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 4 | 408 | `' a'` | latin_word | 35/35 | 5.6 | Stranda hundred. This was an unusual position for a person of her gender in 16th century |
| 5 | 1241 | `' spouse'` | latin_word | 35/35 | 4.6 | borg Castle on several occasions during the absence of her spouse.\n |
| 6 | 1214 | `'.'` | punct_or_symbol | 35/35 | 6.2 | woman, sister of king Gustav I of Sweden. Between 1525 and 15 |
| 7 | 1240 | `' her'` | latin_word | 35/35 | 7.1 | Vyborg Castle on several occasions during the absence of her spouse.\n |
| 8 | 1227 | `','` | punct_or_symbol | 35/35 | 8.5 | 1525 and 1534, she commanded Vyborg Castle on several occasions during the |
| 9 | 1222 | `' '` | space | 35/35 | 8.4 | of Sweden. Between 1525 and 1534, she commanded Vyborg Castle |
| 10 | 1239 | `' of'` | latin_word | 35/35 | 9.7 | commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 11 | 1238 | `' absence'` | latin_word | 35/35 | 11.7 | she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 12 | 1216 | `' '` | space | 35/35 | 14.7 | sister of king Gustav I of Sweden. Between 1525 and 1534 |
| 13 | 1236 | `' during'` | latin_word | 35/35 | 14.1 | 4, she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 14 | 1229 | `' commanded'` | latin_word | 35/35 | 14.4 | 25 and 1534, she commanded Vyborg Castle on several occasions during the absence of |
| 15 | 1233 | `' on'` | latin_word | 35/35 | 16.1 | 1534, she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 16 | 1234 | `' several'` | latin_word | 35/35 | 17.1 | 534, she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 17 | 1228 | `' she'` | latin_word | 35/35 | 17.7 | 525 and 1534, she commanded Vyborg Castle on several occasions during the absence |
| 18 | 1205 | `','` | punct_or_symbol | 35/35 | 19.5 | ta of Hoya", was a Swedish noblewoman, sister of king Gustav I of Sweden. Between |
| 19 | 1232 | `' Castle'` | latin_word | 35/35 | 19.5 | 1534, she commanded Vyborg Castle on several occasions during the absence of her spouse.\n |
| 20 | 1201 | `' a'` | latin_word | 35/35 | 20.0 | and "Margareta of Hoya", was a Swedish noblewoman, sister of king Gustav I |
| 21 | 1221 | `' and'` | latin_word | 35/35 | 20.7 | I of Sweden. Between 1525 and 1534, she commanded Vyborg |
| 22 | 1230 | `' Vy'` | latin_word | 35/35 | 22.2 | 5 and 1534, she commanded Vyborg Castle on several occasions during the absence of her |
| 23 | 1181 | `'),'` | punct_or_symbol | 35/35 | 23.4 | – 31 December 1536), also called "Margareta Vasa" and |
| 24 | 1231 | `'borg'` | latin_word | 35/35 | 24.4 | and 1534, she commanded Vyborg Castle on several occasions during the absence of her spouse |
| 25 | 1199 | `'",'` | punct_or_symbol | 35/35 | 25.9 | asa" and "Margareta of Hoya", was a Swedish noblewoman, sister of king Gust |
| 26 | 1172 | `' '` | space | 35/35 | 26.6 | dotter Vasa (1497 – 31 December 1536), also |
| 27 | 1215 | `' Between'` | latin_word | 35/35 | 29.5 | , sister of king Gustav I of Sweden. Between 1525 and 153 |
| 28 | 1153 | `'.\n'` | punct_or_symbol | 35/35 | 32.6 | wig-Holstein-Sonderburg-Beck.\nDocument: Margareta Eriksdotter |
| 29 | 1217 | `'1'` | number | 35/35 | 33.1 | of king Gustav I of Sweden. Between 1525 and 1534, |
| 30 | 1223 | `'1'` | number | 35/35 | 33.3 | Sweden. Between 1525 and 1534, she commanded Vyborg Castle on |
| 31 | 1220 | `'5'` | number | 35/35 | 34.4 | av I of Sweden. Between 1525 and 1534, she commanded Vy |
| 32 | 1190 | `'"'` | punct_or_symbol | 35/35 | 33.7 | 6), also called "Margareta Vasa" and "Margareta of Hoya", was |
| 33 | 1204 | `'woman'` | latin_word | 35/35 | 35.1 | areta of Hoya", was a Swedish noblewoman, sister of king Gustav I of Sweden. |
| 34 | 1226 | `'4'` | number | 35/35 | 37.8 | 1525 and 1534, she commanded Vyborg Castle on several occasions during |
| 35 | 1200 | `' was'` | latin_word | 35/35 | 38.5 | " and "Margareta of Hoya", was a Swedish noblewoman, sister of king Gustav |
| 36 | 1048 | `','` | punct_or_symbol | 35/35 | 40.4 | ée Bremer. He married Caroline Boeck, daughter of captain Cæsar Boeck. |
| 37 | 1224 | `'5'` | number | 35/35 | 39.6 | . Between 1525 and 1534, she commanded Vyborg Castle on several |
| 38 | 1111 | `' the'` | latin_word | 35/35 | 40.5 | ) was a Danish-German prince and member of the House of Oldenburg. After acquiring the estate of |
| 39 | 1191 | `' and'` | latin_word | 35/35 | 41.3 | ), also called "Margareta Vasa" and "Margareta of Hoya", was a |
| 40 | 1218 | `'5'` | number | 35/35 | 42.5 | king Gustav I of Sweden. Between 1525 and 1534, she |
| 41 | 1134 | `' he'` | latin_word | 35/35 | 48.1 | Westfalen in 1646, he took the title of Duke of Schleswig-H |
| 42 | 1213 | `' Sweden'` | latin_word | 35/35 | 46.3 | noblewoman, sister of king Gustav I of Sweden. Between 1525 and 1 |
| 43 | 1202 | `' Swedish'` | latin_word | 35/35 | 48.5 | "Margareta of Hoya", was a Swedish noblewoman, sister of king Gustav I of |
| 44 | 1207 | `' of'` | latin_word | 35/35 | 47.0 | Hoya", was a Swedish noblewoman, sister of king Gustav I of Sweden. Between 1 |
| 45 | 1116 | `'.'` | punct_or_symbol | 35/35 | 50.1 | erman prince and member of the House of Oldenburg. After acquiring the estate of Beck in Westfalen |
| 46 | 1219 | `'2'` | number | 35/35 | 50.5 | Gustav I of Sweden. Between 1525 and 1534, she commanded |
| 47 | 1225 | `'3'` | number | 35/35 | 50.6 | Between 1525 and 1534, she commanded Vyborg Castle on several occasions |
| 48 | 1212 | `' of'` | latin_word | 35/35 | 51.9 | Swedish noblewoman, sister of king Gustav I of Sweden. Between 1525 and |
| 49 | 1206 | `' sister'` | latin_word | 35/35 | 53.0 | of Hoya", was a Swedish noblewoman, sister of king Gustav I of Sweden. Between |
| 50 | 780 | `' '` | space | 35/35 | 70.2 | ) on the Forbes list with an estimated fortune of 142 billion Danish kroner, which made |
| 51 | 1203 | `' noble'` | latin_word | 35/35 | 55.6 | Margareta of Hoya", was a Swedish noblewoman, sister of king Gustav I of Sweden |
| 52 | 1211 | `' I'` | latin_word | 35/35 | 55.9 | a Swedish noblewoman, sister of king Gustav I of Sweden. Between 1525 and |
| 53 | 1208 | `' king'` | latin_word | 35/35 | 57.2 | oya", was a Swedish noblewoman, sister of king Gustav I of Sweden. Between 15 |
| 54 | 1136 | `' the'` | latin_word | 35/35 | 59.9 | alen in 1646, he took the title of Duke of Schleswig-Holstein |
| 55 | 1119 | `' the'` | latin_word | 35/35 | 59.7 | member of the House of Oldenburg. After acquiring the estate of Beck in Westfalen in 1 |
| 56 | 887 | `' '` | space | 35/35 | 63.2 | dalena Eleonora Busseck. On 26 January 1691, she |
| 57 | 1183 | `' called'` | latin_word | 35/35 | 62.7 | 31 December 1536), also called "Margareta Vasa" and "Marg |
| 58 | 492 | `'ˈ'` | other_text | 35/35 | 63.7 | (neɡolaj kʌsdɐ ˈvaldɑw); born 27 July |
| 59 | 1155 | `':'` | punct_or_symbol | 35/35 | 64.2 | olstein-Sonderburg-Beck.\nDocument: Margareta Eriksdotter Vasa |
| 60 | 1184 | `' "'` | punct_or_symbol | 35/35 | 65.7 | 1 December 1536), also called "Margareta Vasa" and "Margare |
| 61 | 1182 | `' also'` | latin_word | 35/35 | 66.5 | 31 December 1536), also called "Margareta Vasa" and " |
| 62 | 1133 | `','` | punct_or_symbol | 35/35 | 66.9 | in Westfalen in 1646, he took the title of Duke of Schleswig |
| 63 | 299 | `' the'` | latin_word | 35/35 | 68.6 | .\nDocument: Sigrid Sture: She was the daughter of Svante Stensson Sture and |
| 64 | 1041 | `'.'` | punct_or_symbol | 35/35 | 68.4 | and his wife Caroline Kathrine née Bremer. He married Caroline Boeck, daughter of captain |
| 65 | 1166 | `' ('` | punct_or_symbol | 35/35 | 71.4 | Margareta Eriksdotter Vasa (1497 – 31 December |
| 66 | 1209 | `' Gust'` | latin_word | 35/35 | 71.6 | ", was a Swedish noblewoman, sister of king Gustav I of Sweden. Between 152 |
| 67 | 995 | `'.\n'` | punct_or_symbol | 35/35 | 77.5 | ardment of Brussels in 1695.\nDocument: Daniel Bremer Juell was born in |
| 68 | 1198 | `'oya'` | latin_word | 35/35 | 79.5 | Vasa" and "Margareta of Hoya", was a Swedish noblewoman, sister of king |
| 69 | 983 | `' the'` | latin_word | 35/35 | 79.4 | commissions: she gave birth to their youngest sons during the Bombardment of Brussels in 169 |
| 70 | 522 | `' the'` | latin_word | 35/35 | 80.9 | actor, producer and screenwriter. He graduated from the Danish National School of Theatre in Copenhagen in 1 |
| 71 | 673 | `'-school'` | other_text | 35/35 | 79.7 | Møller. Mærsk married his high-school sweetheart, Emma Neergaard Rasmussen |
| 72 | 719 | `'),'` | punct_or_symbol | 35/35 | 81.8 | : Leise (born 1941), Kirsten, Mrs Olufsen (born |
| 73 | 1189 | `'asa'` | latin_word | 35/35 | 84.2 | 36), also called "Margareta Vasa" and "Margareta of Hoya", |
| 74 | 1176 | `' '` | space | 35/35 | 84.1 | (1497 – 31 December 1536), also called "Margare |
| 75 | 1210 | `'av'` | latin_word | 35/35 | 84.4 | was a Swedish noblewoman, sister of king Gustav I of Sweden. Between 1525 |
| 76 | 379 | `' the'` | latin_word | 35/35 | 92.9 | Svante Turesson Bielke. After the death of her husband in 1577 |
| 77 | 1196 | `' of'` | latin_word | 35/35 | 87.1 | areta Vasa" and "Margareta of Hoya", was a Swedish noblewoman, sister |
| 78 | 1063 | `'.\n'` | punct_or_symbol | 35/35 | 88.3 | æsar Boeck. They had three children.\nDocument: August Philipp, Duke of Schleswig |
| 79 | 763 | `' the'` | latin_word | 35/35 | 88.5 | 557th-wealthiest person in the world (2007) on the Forbes |
| 80 | 1167 | `'1'` | number | 35/35 | 93.7 | areta Eriksdotter Vasa (1497 – 31 December 1 |

## Example 4

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2604 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | convinced Richard to allow John into England in his absence.\n |
| 2 | 2587 | `'.'` | punct_or_symbol | 35/35 | 2.4 | which would prove to be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow |
| 3 | 2603 | `' absence'` | latin_word | 35/35 | 3.0 | , convinced Richard to allow John into England in his absence.\n |
| 4 | 2602 | `' his'` | latin_word | 35/35 | 4.7 | mother, convinced Richard to allow John into England in his absence.\n |
| 5 | 2601 | `' in'` | latin_word | 35/35 | 5.5 | queen mother, convinced Richard to allow John into England in his absence.\n |
| 6 | 2589 | `','` | punct_or_symbol | 35/35 | 7.7 | prove to be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into |
| 7 | 2539 | `','` | punct_or_symbol | 35/35 | 10.1 | Hugh de Puiset and William Mandeville, and made William Longchamp, the Bishop of |
| 8 | 2593 | `','` | punct_or_symbol | 35/35 | 9.5 | less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into England in his absence |
| 9 | 2599 | `' into'` | latin_word | 35/35 | 9.5 | , the queen mother, convinced Richard to allow John into England in his absence.\n |
| 10 | 2590 | `' the'` | latin_word | 35/35 | 10.6 | to be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into England |
| 11 | 2470 | `' return'` | latin_word | 35/35 | 13.0 | of Brittany as the heir to the throne. In return, John promised not to visit England for the next |
| 12 | 2600 | `' England'` | latin_word | 35/35 | 12.5 | the queen mother, convinced Richard to allow John into England in his absence.\n |
| 13 | 2582 | `' a'` | latin_word | 35/35 | 16.3 | with Puiset, which would prove to be a less than satisfactory partnership. Eleanor, the queen mother |
| 14 | 2597 | `' allow'` | latin_word | 35/35 | 14.9 | . Eleanor, the queen mother, convinced Richard to allow John into England in his absence.\n |
| 15 | 2596 | `' to'` | latin_word | 35/35 | 15.3 | partnership. Eleanor, the queen mother, convinced Richard to allow John into England in his absence.\n |
| 16 | 2484 | `' thereby'` | latin_word | 35/35 | 22.2 | not to visit England for the next three years, thereby in theory giving Richard adequate time to conduct a successful |
| 17 | 2594 | `' convinced'` | latin_word | 35/35 | 15.7 | than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into England in his absence.\n |
| 18 | 2598 | `' John'` | latin_word | 35/35 | 17.5 | Eleanor, the queen mother, convinced Richard to allow John into England in his absence.\n |
| 19 | 2580 | `' to'` | latin_word | 35/35 | 19.7 | justiciar with Puiset, which would prove to be a less than satisfactory partnership. Eleanor, the |
| 20 | 2588 | `' Eleanor'` | latin_word | 35/35 | 20.0 | would prove to be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John |
| 21 | 2595 | `' Richard'` | latin_word | 35/35 | 21.2 | satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into England in his absence.\n |
| 22 | 2586 | `' partnership'` | latin_word | 35/35 | 22.5 | , which would prove to be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to |
| 23 | 2576 | `','` | punct_or_symbol | 35/35 | 23.1 | took over as joint justiciar with Puiset, which would prove to be a less than satisfactory partnership |
| 24 | 2555 | `'.'` | punct_or_symbol | 35/35 | 23.6 | amp, the Bishop of Ely, his chancellor. Mandeville immediately died, and Longchamp |
| 25 | 2591 | `' queen'` | latin_word | 35/35 | 24.9 | be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into England in |
| 26 | 2592 | `' mother'` | latin_word | 35/35 | 25.5 | a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard to allow John into England in his |
| 27 | 1197 | `' their'` | latin_word | 35/35 | 50.4 | The Anglo-Saxon monarchs used various locations for their coronations, including Bath, Kingston upon Thames, |
| 28 | 1683 | `' to'` | latin_word | 35/35 | 42.3 | fought successive, increasingly expensive, campaigns in a bid to regain these possessions. John's efforts to raise revenues |
| 29 | 2585 | `' satisfactory'` | latin_word | 35/35 | 33.1 | et, which would prove to be a less than satisfactory partnership. Eleanor, the queen mother, convinced Richard |
| 30 | 2561 | `','` | punct_or_symbol | 35/35 | 34.8 | ly, his chancellor. Mandeville immediately died, and Longchamp took over as joint justiciar |
| 31 | 2579 | `' prove'` | latin_word | 35/35 | 34.6 | joint justiciar with Puiset, which would prove to be a less than satisfactory partnership. Eleanor, |
| 32 | 2584 | `' than'` | latin_word | 35/35 | 35.3 | uiset, which would prove to be a less than satisfactory partnership. Eleanor, the queen mother, convinced |
| 33 | 2578 | `' would'` | latin_word | 35/35 | 37.5 | as joint justiciar with Puiset, which would prove to be a less than satisfactory partnership. Eleanor |
| 34 | 2577 | `' which'` | latin_word | 35/35 | 40.6 | over as joint justiciar with Puiset, which would prove to be a less than satisfactory partnership. |
| 35 | 2581 | `' be'` | latin_word | 35/35 | 41.0 | iciar with Puiset, which would prove to be a less than satisfactory partnership. Eleanor, the queen |
| 36 | 2583 | `' less'` | latin_word | 35/35 | 42.0 | Puiset, which would prove to be a less than satisfactory partnership. Eleanor, the queen mother, |
| 37 | 2029 | `' the'` | latin_word | 35/35 | 47.3 | stalemate and a tense family reconciliation in England at the end of 1184.\nDocument: |
| 38 | 2571 | `'iciar'` | latin_word | 35/35 | 46.1 | , and Longchamp took over as joint justiciar with Puiset, which would prove to be |
| 39 | 2205 | `' English'` | latin_word | 35/35 | 56.0 | 40.\nDocument: Katherine of England (Old English: "Katerine"; 25 November |
| 40 | 2572 | `' with'` | latin_word | 35/35 | 42.6 | and Longchamp took over as joint justiciar with Puiset, which would prove to be a |
| 41 | 2552 | `','` | punct_or_symbol | 35/35 | 46.1 | William Longchamp, the Bishop of Ely, his chancellor. Mandeville immediately died, and |
| 42 | 2468 | `'.'` | punct_or_symbol | 35/35 | 44.3 | -old Arthur of Brittany as the heir to the throne. In return, John promised not to visit England for |
| 43 | 2546 | `','` | punct_or_symbol | 35/35 | 45.6 | Mandeville, and made William Longchamp, the Bishop of Ely, his chancellor. Mand |
| 44 | 2479 | `' the'` | latin_word | 35/35 | 54.8 | In return, John promised not to visit England for the next three years, thereby in theory giving Richard adequate |
| 45 | 2509 | `'.'` | punct_or_symbol | 35/35 | 49.1 | from the Levant without fear of John seizing power. Richard left political authority in England – the post of |
| 46 | 2566 | `' took'` | latin_word | 35/35 | 50.6 | Mandeville immediately died, and Longchamp took over as joint justiciar with Puiset, |
| 47 | 2568 | `' as'` | latin_word | 35/35 | 51.6 | ille immediately died, and Longchamp took over as joint justiciar with Puiset, which would |
| 48 | 977 | `' '` | space | 35/35 | 59.6 | 1540 – 26 January 1568), born Lady Katherine Grey, |
| 49 | 2230 | `' the'` | latin_word | 35/35 | 61.3 | 33 May 1257) was the fifth child of Henry III and his wife, Eleanor |
| 50 | 2567 | `' over'` | latin_word | 35/35 | 54.8 | eville immediately died, and Longchamp took over as joint justiciar with Puiset, which |
| 51 | 2562 | `' and'` | latin_word | 35/35 | 56.5 | , his chancellor. Mandeville immediately died, and Longchamp took over as joint justiciar with |
| 52 | 2570 | `' just'` | latin_word | 35/35 | 55.7 | died, and Longchamp took over as joint justiciar with Puiset, which would prove to |
| 53 | 2525 | `' the'` | latin_word | 35/35 | 58.8 | England – the post of justiciar – jointly in the hands of Bishop Hugh de Puiset and William |
| 54 | 2569 | `' joint'` | latin_word | 35/35 | 60.2 | immediately died, and Longchamp took over as joint justiciar with Puiset, which would prove |
| 55 | 2547 | `' the'` | latin_word | 35/35 | 60.2 | eville, and made William Longchamp, the Bishop of Ely, his chancellor. Mandev |
| 56 | 2573 | `' P'` | latin_word | 35/35 | 62.3 | Longchamp took over as joint justiciar with Puiset, which would prove to be a less |
| 57 | 2574 | `'uis'` | latin_word | 35/35 | 62.7 | champ took over as joint justiciar with Puiset, which would prove to be a less than |
| 58 | 2556 | `' Mand'` | latin_word | 35/35 | 66.1 | , the Bishop of Ely, his chancellor. Mandeville immediately died, and Longchamp took |
| 59 | 2500 | `' the'` | latin_word | 35/35 | 63.1 | time to conduct a successful crusade and return from the Levant without fear of John seizing power. Richard |
| 60 | 2560 | `' died'` | latin_word | 35/35 | 64.9 | Ely, his chancellor. Mandeville immediately died, and Longchamp took over as joint just |
| 61 | 2575 | `'et'` | latin_word | 35/35 | 66.9 | amp took over as joint justiciar with Puiset, which would prove to be a less than satisfactory |
| 62 | 440 | `' '` | space | 35/35 | 91.3 | eliza or Adelida (died before 1113) was a daughter of the |
| 63 | 2185 | `' the'` | latin_word | 35/35 | 73.8 | 20th century. Westminster suffered minor damage during the Blitz on 15 November 194 |
| 64 | 2175 | `'2'` | number | 35/35 | 94.9 | English Bible was also put together here in the 20th century. Westminster suffered minor damage during the |
| 65 | 2565 | `'amp'` | latin_word | 35/35 | 74.1 | . Mandeville immediately died, and Longchamp took over as joint justiciar with Puiset |
| 66 | 2176 | `'0'` | number | 35/35 | 79.9 | Bible was also put together here in the 20th century. Westminster suffered minor damage during the Blitz |
| 67 | 2210 | `'ine'` | latin_word | 35/35 | 82.9 | Katherine of England (Old English: "Katerine"; 25 November 1253 |
| 68 | 2506 | `' John'` | latin_word | 35/35 | 100.6 | ade and return from the Levant without fear of John seizing power. Richard left political authority in England – |
| 69 | 2486 | `' theory'` | latin_word | 35/35 | 88.7 | visit England for the next three years, thereby in theory giving Richard adequate time to conduct a successful crusade |
| 70 | 2559 | `' immediately'` | latin_word | 35/35 | 74.0 | of Ely, his chancellor. Mandeville immediately died, and Longchamp took over as joint |
| 71 | 2519 | `' of'` | latin_word | 35/35 | 97.7 | . Richard left political authority in England – the post of justiciar – jointly in the hands of Bishop Hugh |
| 72 | 1088 | `' '` | space | 35/35 | 84.5 | Alice of Armenia (1182 – after 1234) was ruling Lady of Tor |
| 73 | 2534 | `' and'` | latin_word | 35/35 | 76.8 | in the hands of Bishop Hugh de Puiset and William Mandeville, and made William Longch |
| 74 | 2432 | `','` | punct_or_symbol | 35/35 | 78.5 | retained royal control of key castles in these counties, thereby preventing John from accumulating too much military and political |
| 75 | 2554 | `' chancellor'` | latin_word | 35/35 | 79.7 | champ, the Bishop of Ely, his chancellor. Mandeville immediately died, and Longch |
| 76 | 2517 | `' the'` | latin_word | 35/35 | 85.4 | seizing power. Richard left political authority in England – the post of justiciar – jointly in the hands of |
| 77 | 2553 | `' his'` | latin_word | 35/35 | 85.4 | Longchamp, the Bishop of Ely, his chancellor. Mandeville immediately died, and Long |
| 78 | 1407 | `' was'` | latin_word | 35/35 | 99.6 | taken control of the city, and so the king was crowned in Gloucester Cathedral. This coronation was |
| 79 | 2540 | `' and'` | latin_word | 35/35 | 87.1 | de Puiset and William Mandeville, and made William Longchamp, the Bishop of E |
| 80 | 2360 | `'.'` | punct_or_symbol | 35/35 | 90.3 | would not face a revolt while away from his empire. John was made Count of Mortain, was married |

## Example 5

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2726 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | of the Client–server model helped to launch Ethernet.\n |
| 2 | 2725 | `' Ethernet'` | latin_word | 35/35 | 2.0 | vision of the Client–server model helped to launch Ethernet.\n |
| 3 | 2717 | `' the'` | latin_word | 35/35 | 4.1 | of the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 4 | 2724 | `' launch'` | latin_word | 35/35 | 4.6 | futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 5 | 2694 | `'2'` | number | 35/35 | 7.3 | and missed the opportunity to fund Lotus 1-2-3 or Visicalc. He also missed |
| 6 | 2705 | `' the'` | latin_word | 35/35 | 7.4 | -3 or Visicalc. He also missed the importance of the personal computer, but his futuristic vision |
| 7 | 2723 | `' to'` | latin_word | 35/35 | 7.2 | his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 8 | 2722 | `' helped'` | latin_word | 35/35 | 8.6 | but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 9 | 1071 | `' the'` | latin_word | 35/35 | 11.7 | 007 and tested on millions of viewers of the leading satellite TV operator, Tricolor TV (over |
| 10 | 2691 | `' '` | space | 35/35 | 12.1 | from hardware, and missed the opportunity to fund Lotus 1-2-3 or Visicalc. |
| 11 | 2711 | `','` | punct_or_symbol | 35/35 | 12.2 | . He also missed the importance of the personal computer, but his futuristic vision of the Client–server model |
| 12 | 2708 | `' the'` | latin_word | 35/35 | 12.1 | Visicalc. He also missed the importance of the personal computer, but his futuristic vision of the Client |
| 13 | 2719 | `'–'` | punct_or_symbol | 35/35 | 13.5 | personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 14 | 2701 | `'.'` | punct_or_symbol | 35/35 | 13.3 | 1-2-3 or Visicalc. He also missed the importance of the personal computer, |
| 15 | 2721 | `' model'` | latin_word | 35/35 | 16.5 | , but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 16 | 2718 | `' Client'` | latin_word | 35/35 | 19.0 | the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 17 | 2720 | `'server'` | latin_word | 35/35 | 19.7 | computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 18 | 2714 | `' futuristic'` | latin_word | 35/35 | 21.1 | missed the importance of the personal computer, but his futuristic vision of the Client–server model helped to launch |
| 19 | 2713 | `' his'` | latin_word | 35/35 | 21.3 | also missed the importance of the personal computer, but his futuristic vision of the Client–server model helped to |
| 20 | 2613 | `' the'` | latin_word | 35/35 | 23.5 | of his own founding, Digital Equipment Corporation. At the time the book was published by two computer journal writers |
| 21 | 2715 | `' vision'` | latin_word | 35/35 | 22.8 | the importance of the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet |
| 22 | 2666 | `' the'` | latin_word | 35/35 | 25.3 | Development Corporation, and Apollo Computer. While believing in the value of software, he did not believe in the |
| 23 | 2120 | `'2'` | number | 35/35 | 36.5 | 2Dell blog, and then in February 2007, Michael Dell launched IdeaStorm.com |
| 24 | 2480 | `' '` | space | 35/35 | 30.6 | The PA-8000 was introduced on 2 November 1995 when shipments began |
| 25 | 2716 | `' of'` | latin_word | 35/35 | 26.0 | importance of the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 26 | 2712 | `' but'` | latin_word | 35/35 | 27.6 | He also missed the importance of the personal computer, but his futuristic vision of the Client–server model helped |
| 27 | 2703 | `' also'` | latin_word | 35/35 | 28.7 | -2-3 or Visicalc. He also missed the importance of the personal computer, but his |
| 28 | 2702 | `' He'` | latin_word | 35/35 | 29.5 | 1-2-3 or Visicalc. He also missed the importance of the personal computer, but |
| 29 | 2676 | `' the'` | latin_word | 35/35 | 30.8 | the value of software, he did not believe in the value of software separate from hardware, and missed the |
| 30 | 2704 | `' missed'` | latin_word | 35/35 | 31.6 | 2-3 or Visicalc. He also missed the importance of the personal computer, but his futuristic |
| 31 | 1818 | `'6'` | number | 35/35 | 41.2 | expensive Macintosh systems. The performance advantage of 68000-based Macintosh systems was er |
| 32 | 2710 | `' computer'` | latin_word | 35/35 | 33.0 | c. He also missed the importance of the personal computer, but his futuristic vision of the Client–server |
| 33 | 2686 | `' the'` | latin_word | 35/35 | 33.9 | the value of software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 |
| 34 | 2168 | `' '` | space | 35/35 | 62.9 | the negative blog posts from 49% to 22%, as well as reduce the "D |
| 35 | 2709 | `' personal'` | latin_word | 35/35 | 35.9 | icalc. He also missed the importance of the personal computer, but his futuristic vision of the Client– |
| 36 | 2706 | `' importance'` | latin_word | 35/35 | 36.1 | 3 or Visicalc. He also missed the importance of the personal computer, but his futuristic vision of |
| 37 | 2211 | `' the'` | latin_word | 35/35 | 38.7 | acquire the assets of the largest cable television operator at the time, AT&T Broadband, for US$ |
| 38 | 1819 | `'8'` | number | 35/35 | 60.9 | Macintosh systems. The performance advantage of 68000-based Macintosh systems was eroded |
| 39 | 2662 | `'.'` | punct_or_symbol | 35/35 | 39.1 | Symbolics, Lotus Development Corporation, and Apollo Computer. While believing in the value of software, he did |
| 40 | 2707 | `' of'` | latin_word | 35/35 | 40.3 | or Visicalc. He also missed the importance of the personal computer, but his futuristic vision of the |
| 41 | 2684 | `' and'` | latin_word | 35/35 | 41.1 | believe in the value of software separate from hardware, and missed the opportunity to fund Lotus 1-2 |
| 42 | 2695 | `'-'` | punct_or_symbol | 35/35 | 44.8 | missed the opportunity to fund Lotus 1-2-3 or Visicalc. He also missed the |
| 43 | 2683 | `','` | punct_or_symbol | 35/35 | 42.6 | not believe in the value of software separate from hardware, and missed the opportunity to fund Lotus 1- |
| 44 | 2591 | `' Ken'` | latin_word | 35/35 | 53.6 | and Digital Equipment Corporation, chronicles the experiences of Ken Olsen racing to design minicomputers at the |
| 45 | 2693 | `'-'` | punct_or_symbol | 35/35 | 44.3 | , and missed the opportunity to fund Lotus 1-2-3 or Visicalc. He also |
| 46 | 2670 | `','` | punct_or_symbol | 35/35 | 44.5 | Apollo Computer. While believing in the value of software, he did not believe in the value of software separate |
| 47 | 2697 | `' or'` | latin_word | 35/35 | 45.0 | opportunity to fund Lotus 1-2-3 or Visicalc. He also missed the importance of |
| 48 | 2413 | `' code'` | latin_word | 35/35 | 50.5 | -8000 (PCX-U), code-named "Onyx", is a microprocessor |
| 49 | 2699 | `'ical'` | latin_word | 35/35 | 51.9 | fund Lotus 1-2-3 or Visicalc. He also missed the importance of the personal |
| 50 | 2698 | `' Vis'` | latin_word | 35/35 | 52.9 | to fund Lotus 1-2-3 or Visicalc. He also missed the importance of the |
| 51 | 2689 | `' fund'` | latin_word | 35/35 | 52.7 | software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or Visical |
| 52 | 2692 | `'1'` | number | 35/35 | 57.6 | hardware, and missed the opportunity to fund Lotus 1-2-3 or Visicalc. He |
| 53 | 2601 | `' the'` | latin_word | 35/35 | 60.1 | Ken Olsen racing to design minicomputers at the company of his own founding, Digital Equipment Corporation. |
| 54 | 2700 | `'c'` | latin_word | 35/35 | 57.7 | Lotus 1-2-3 or Visicalc. He also missed the importance of the personal computer |
| 55 | 2688 | `' to'` | latin_word | 35/35 | 62.2 | of software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or Vis |
| 56 | 2671 | `' he'` | latin_word | 35/35 | 59.1 | Computer. While believing in the value of software, he did not believe in the value of software separate from |
| 57 | 2690 | `' Lotus'` | latin_word | 35/35 | 60.1 | separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or Visicalc |
| 58 | 2696 | `'3'` | number | 35/35 | 61.9 | the opportunity to fund Lotus 1-2-3 or Visicalc. He also missed the importance |
| 59 | 1319 | `'2'` | number | 35/35 | 78.6 | 6 percent at Cisco. In 2002, when Compaq merged with Hewlett Pack |
| 60 | 2611 | `'.'` | punct_or_symbol | 35/35 | 62.0 | the company of his own founding, Digital Equipment Corporation. At the time the book was published by two computer |
| 61 | 2674 | `' believe'` | latin_word | 35/35 | 69.7 | believing in the value of software, he did not believe in the value of software separate from hardware, and |
| 62 | 2664 | `' believing'` | latin_word | 35/35 | 69.9 | , Lotus Development Corporation, and Apollo Computer. While believing in the value of software, he did not believe |
| 63 | 1117 | `','` | punct_or_symbol | 35/35 | 80.1 | of 25 systems that served 80,000 customers in rural areas that Time Warner |
| 64 | 2645 | `'),'` | punct_or_symbol | 35/35 | 72.1 | as Data General (founded by his former employee), Prime Computer, Wang Laboratories, Symbolics, Lotus |
| 65 | 1664 | `'9'` | number | 35/35 | 80.8 | separate MS-DOS and Windows products. Windows 95 significantly enhanced the multimedia capability and performance of IBM |
| 66 | 2618 | `' published'` | latin_word | 35/35 | 111.0 | Digital Equipment Corporation. At the time the book was published by two computer journal writers, Ken Olsen was competing |
| 67 | 2607 | `','` | punct_or_symbol | 35/35 | 82.1 | icomputers at the company of his own founding, Digital Equipment Corporation. At the time the book was |
| 68 | 1801 | `' gradually'` | latin_word | 35/35 | 82.5 | 3.0, then Windows 95, gradually took market share from the more expensive Macintosh systems |
| 69 | 2627 | `' was'` | latin_word | 35/35 | 83.8 | was published by two computer journal writers, Ken Olsen was competing with other Massachusetts computing companies such as Data General |
| 70 | 2663 | `' While'` | latin_word | 35/35 | 77.9 | ics, Lotus Development Corporation, and Apollo Computer. While believing in the value of software, he did not |
| 71 | 1563 | `','` | punct_or_symbol | 35/35 | 84.1 | Acer.\nDocument: Macintosh: Compaq, who had previously held the third place spot among PC |
| 72 | 2412 | `'),'` | punct_or_symbol | 35/35 | 81.0 | PA-8000 (PCX-U), code-named "Onyx", is a micro |
| 73 | 2687 | `' opportunity'` | latin_word | 35/35 | 80.2 | value of software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or |
| 74 | 294 | `' re'` | latin_word | 35/35 | 93.9 | from December 1931 onward which was reissued on the Columbia label as well as the Vocal |
| 75 | 2624 | `','` | punct_or_symbol | 35/35 | 81.7 | time the book was published by two computer journal writers, Ken Olsen was competing with other Massachusetts computing companies such |
| 76 | 2685 | `' missed'` | latin_word | 35/35 | 82.3 | in the value of software separate from hardware, and missed the opportunity to fund Lotus 1-2- |
| 77 | 2669 | `' software'` | latin_word | 35/35 | 83.5 | and Apollo Computer. While believing in the value of software, he did not believe in the value of software |
| 78 | 2560 | `'.\n'` | punct_or_symbol | 35/35 | 83.8 | the basic PA-8000 processor core.\nDocument: The Ultimate Entrepreneur: The biographical book |
| 79 | 2658 | `','` | punct_or_symbol | 35/35 | 86.0 | , Wang Laboratories, Symbolics, Lotus Development Corporation, and Apollo Computer. While believing in the value of |
| 80 | 2672 | `' did'` | latin_word | 35/35 | 86.2 | . While believing in the value of software, he did not believe in the value of software separate from hardware |

## Example 6

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1403 | `'.\n'` | punct_or_symbol | 34/34 | 1.0 | in biology, some of which are relatively well known.\n |
| 2 | 1395 | `','` | punct_or_symbol | 34/34 | 2.8 | ) in Vienna. He wrote systematic works in biology, some of which are relatively well known.\n |
| 3 | 1402 | `' known'` | latin_word | 34/34 | 3.5 | works in biology, some of which are relatively well known.\n |
| 4 | 1388 | `'.'` | punct_or_symbol | 34/34 | 4.5 | gymnasium (secondary school) in Vienna. He wrote systematic works in biology, some of which |
| 5 | 1374 | `' the'` | latin_word | 34/34 | 6.9 | Edler von Hayek, taught natural sciences at the Imperial Realobergymnasium (secondary school |
| 6 | 542 | `'.'` | punct_or_symbol | 34/34 | 10.4 | base of natural logarithms, e = 2.71828...), and found that |
| 7 | 1398 | `' which'` | latin_word | 34/34 | 8.7 | . He wrote systematic works in biology, some of which are relatively well known.\n |
| 8 | 1399 | `' are'` | latin_word | 34/34 | 8.8 | He wrote systematic works in biology, some of which are relatively well known.\n |
| 9 | 1401 | `' well'` | latin_word | 34/34 | 10.5 | systematic works in biology, some of which are relatively well known.\n |
| 10 | 1164 | `' '` | space | 34/34 | 15.5 | z was born in Evanston, Illinois on May 24, 1915. He |
| 11 | 1385 | `')'` | punct_or_symbol | 34/34 | 10.8 | Imperial Realobergymnasium (secondary school) in Vienna. He wrote systematic works in biology, |
| 12 | 126 | `'5'` | number | 34/34 | 16.1 | eventually published as The Sensory Order (1952). It located connective learning at the physical |
| 13 | 1400 | `' relatively'` | latin_word | 34/34 | 13.2 | wrote systematic works in biology, some of which are relatively well known.\n |
| 14 | 1213 | `' a'` | latin_word | 34/34 | 17.4 | 1936, Helmholz won a fellowship to study at the Cambridge University for one year |
| 15 | 1390 | `' wrote'` | latin_word | 34/34 | 15.1 | nasium (secondary school) in Vienna. He wrote systematic works in biology, some of which are relatively |
| 16 | 1218 | `' the'` | latin_word | 34/34 | 18.2 | , Helmholz won a fellowship to study at the Cambridge University for one year. On the advice of |
| 17 | 1391 | `' systematic'` | latin_word | 34/34 | 17.1 | ium (secondary school) in Vienna. He wrote systematic works in biology, some of which are relatively well |
| 18 | 1382 | `' ('` | punct_or_symbol | 34/34 | 18.5 | sciences at the Imperial Realobergymnasium (secondary school) in Vienna. He wrote systematic works |
| 19 | 1394 | `' biology'` | latin_word | 34/34 | 18.1 | school) in Vienna. He wrote systematic works in biology, some of which are relatively well known.\n |
| 20 | 1259 | `' Hay'` | latin_word | 34/34 | 20.6 | , Berkeley for his graduate education.\nDocument: Friedrich Hayek: His father's career as a university professor |
| 21 | 1396 | `' some'` | latin_word | 34/34 | 20.4 | in Vienna. He wrote systematic works in biology, some of which are relatively well known.\n |
| 22 | 1393 | `' in'` | latin_word | 34/34 | 21.2 | secondary school) in Vienna. He wrote systematic works in biology, some of which are relatively well known.\n |
| 23 | 1392 | `' works'` | latin_word | 34/34 | 21.2 | (secondary school) in Vienna. He wrote systematic works in biology, some of which are relatively well known |
| 24 | 1397 | `' of'` | latin_word | 34/34 | 21.7 | Vienna. He wrote systematic works in biology, some of which are relatively well known.\n |
| 25 | 1356 | `'.'` | punct_or_symbol | 34/34 | 24.0 | statistician and was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von |
| 26 | 1389 | `' He'` | latin_word | 34/34 | 23.9 | ymnasium (secondary school) in Vienna. He wrote systematic works in biology, some of which are |
| 27 | 1327 | `','` | punct_or_symbol | 34/34 | 24.9 | of Eugen Böhm von Bawerk, one of the founders of the Austrian School of Economics |
| 28 | 1168 | `' '` | space | 34/34 | 27.1 | Evanston, Illinois on May 24, 1915. He attended the Shatt |
| 29 | 1353 | `' the'` | latin_word | 34/34 | 25.3 | ek was a statistician and was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav |
| 30 | 1369 | `','` | punct_or_symbol | 34/34 | 28.9 | paternal grandfather, Gustav Edler von Hayek, taught natural sciences at the Imperial Realobergym |
| 31 | 1361 | `','` | punct_or_symbol | 34/34 | 31.4 | employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek, taught natural |
| 32 | 1386 | `' in'` | latin_word | 34/34 | 32.3 | Realobergymnasium (secondary school) in Vienna. He wrote systematic works in biology, some |
| 33 | 1338 | `'.'` | punct_or_symbol | 34/34 | 33.8 | one of the founders of the Austrian School of Economics. Von Juraschek was a statistician and |
| 34 | 1387 | `' Vienna'` | latin_word | 34/34 | 35.6 | obergymnasium (secondary school) in Vienna. He wrote systematic works in biology, some of |
| 35 | 1333 | `' the'` | latin_word | 34/34 | 39.2 | von Bawerk, one of the founders of the Austrian School of Economics. Von Juraschek |
| 36 | 1383 | `'secondary'` | latin_word | 34/34 | 37.5 | at the Imperial Realobergymnasium (secondary school) in Vienna. He wrote systematic works in |
| 37 | 1330 | `' the'` | latin_word | 34/34 | 40.5 | Böhm von Bawerk, one of the founders of the Austrian School of Economics. Von J |
| 38 | 1380 | `'nas'` | latin_word | 34/34 | 41.0 | taught natural sciences at the Imperial Realobergymnasium (secondary school) in Vienna. He wrote |
| 39 | 1357 | `' Friedrich'` | latin_word | 34/34 | 44.6 | ian and was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hay |
| 40 | 1370 | `' taught'` | latin_word | 34/34 | 42.2 | grandfather, Gustav Edler von Hayek, taught natural sciences at the Imperial Realobergymnas |
| 41 | 1384 | `' school'` | latin_word | 34/34 | 41.9 | the Imperial Realobergymnasium (secondary school) in Vienna. He wrote systematic works in biology |
| 42 | 1348 | `' and'` | latin_word | 34/34 | 44.1 | . Von Juraschek was a statistician and was later employed by the Austrian government. Friedrich's |
| 43 | 1345 | `' a'` | latin_word | 34/34 | 45.4 | School of Economics. Von Juraschek was a statistician and was later employed by the Austrian government |
| 44 | 1375 | `' Imperial'` | latin_word | 34/34 | 45.9 | ler von Hayek, taught natural sciences at the Imperial Realobergymnasium (secondary school) |
| 45 | 1381 | `'ium'` | latin_word | 34/34 | 45.9 | natural sciences at the Imperial Realobergymnasium (secondary school) in Vienna. He wrote systematic |
| 46 | 1360 | `' grandfather'` | latin_word | 34/34 | 47.2 | later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek, taught |
| 47 | 1377 | `'ober'` | latin_word | 34/34 | 48.1 | Hayek, taught natural sciences at the Imperial Realobergymnasium (secondary school) in Vienna |
| 48 | 1276 | `' life'` | latin_word | 34/34 | 54.3 | as a university professor influenced Friedrich's goals later in life. Both of his grandfathers, who lived |
| 49 | 1373 | `' at'` | latin_word | 34/34 | 50.4 | av Edler von Hayek, taught natural sciences at the Imperial Realobergymnasium (secondary |
| 50 | 1372 | `' sciences'` | latin_word | 34/34 | 51.1 | Gustav Edler von Hayek, taught natural sciences at the Imperial Realobergymnasium ( |
| 51 | 1176 | `' the'` | latin_word | 34/34 | 56.7 | 4, 1915. He attended the Shattuck School military academy in Faribault |
| 52 | 1371 | `' natural'` | latin_word | 34/34 | 52.8 | , Gustav Edler von Hayek, taught natural sciences at the Imperial Realobergymnasium |
| 53 | 1379 | `'ym'` | latin_word | 34/34 | 53.8 | , taught natural sciences at the Imperial Realobergymnasium (secondary school) in Vienna. He |
| 54 | 1344 | `' was'` | latin_word | 34/34 | 57.5 | Austrian School of Economics. Von Juraschek was a statistician and was later employed by the Austrian |
| 55 | 1346 | `' statistic'` | latin_word | 34/34 | 58.4 | of Economics. Von Juraschek was a statistician and was later employed by the Austrian government. |
| 56 | 1376 | `' Real'` | latin_word | 34/34 | 58.8 | von Hayek, taught natural sciences at the Imperial Realobergymnasium (secondary school) in |
| 57 | 1177 | `' Sh'` | latin_word | 34/34 | 60.9 | , 1915. He attended the Shattuck School military academy in Faribault, |
| 58 | 1281 | `' grand'` | latin_word | 34/34 | 74.1 | Friedrich's goals later in life. Both of his grandfathers, who lived long enough for Friedrich to |
| 59 | 1277 | `'.'` | punct_or_symbol | 34/34 | 60.9 | a university professor influenced Friedrich's goals later in life. Both of his grandfathers, who lived long |
| 60 | 1355 | `' government'` | latin_word | 34/34 | 62.4 | a statistician and was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler |
| 61 | 1358 | `"'s"` | other_text | 34/34 | 61.9 | and was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek |
| 62 | 1297 | `'.'` | punct_or_symbol | 34/34 | 64.9 | long enough for Friedrich to know them, were scholars. Franz von Juraschek was a leading economist |
| 63 | 1305 | `' a'` | latin_word | 34/34 | 69.8 | were scholars. Franz von Juraschek was a leading economist in Austria-Hungary and a close |
| 64 | 1378 | `'g'` | latin_word | 34/34 | 66.4 | ek, taught natural sciences at the Imperial Realobergymnasium (secondary school) in Vienna. |
| 65 | 1352 | `' by'` | latin_word | 34/34 | 66.9 | chek was a statistician and was later employed by the Austrian government. Friedrich's paternal grandfather, Gust |
| 66 | 1365 | `'ler'` | latin_word | 34/34 | 68.5 | government. Friedrich's paternal grandfather, Gustav Edler von Hayek, taught natural sciences at the Imperial |
| 67 | 1261 | `':'` | punct_or_symbol | 34/34 | 69.4 | for his graduate education.\nDocument: Friedrich Hayek: His father's career as a university professor influenced Friedrich |
| 68 | 1362 | `' Gust'` | latin_word | 34/34 | 71.0 | by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek, taught natural sciences |
| 69 | 1366 | `' von'` | latin_word | 34/34 | 72.1 | . Friedrich's paternal grandfather, Gustav Edler von Hayek, taught natural sciences at the Imperial Real |
| 70 | 1255 | `'.\n'` | punct_or_symbol | 34/34 | 73.1 | the University of California, Berkeley for his graduate education.\nDocument: Friedrich Hayek: His father's career |
| 71 | 825 | `','` | punct_or_symbol | 34/34 | 74.7 | an offer from the Institute for Advanced Study in Princeton, New Jersey, despite the presence there of such distinguished |
| 72 | 1142 | `'.\n'` | punct_or_symbol | 34/34 | 74.3 | American economy and the operations of the US Federal Reserve.\nDocument: A. Carl Helmholz: Helm |
| 73 | 1359 | `' paternal'` | latin_word | 34/34 | 74.9 | was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek, |
| 74 | 1349 | `' was'` | latin_word | 34/34 | 76.4 | Von Juraschek was a statistician and was later employed by the Austrian government. Friedrich's paternal |
| 75 | 1364 | `' Ed'` | latin_word | 34/34 | 76.9 | Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek, taught natural sciences at the |
| 76 | 1347 | `'ian'` | latin_word | 34/34 | 80.0 | Economics. Von Juraschek was a statistician and was later employed by the Austrian government. Friedrich |
| 77 | 808 | `' the'` | latin_word | 34/34 | 83.0 | system.\nDocument: Richard Feynman: After the war, Feynman declined an offer from the |
| 78 | 1368 | `'ek'` | latin_word | 34/34 | 81.2 | 's paternal grandfather, Gustav Edler von Hayek, taught natural sciences at the Imperial Realoberg |
| 79 | 1367 | `' Hay'` | latin_word | 34/34 | 82.8 | Friedrich's paternal grandfather, Gustav Edler von Hayek, taught natural sciences at the Imperial Realober |
| 80 | 1201 | `'.'` | punct_or_symbol | 34/34 | 83.9 | which he went to Harvard University for his undergraduate education. In 1936, Helmholz |

## Example 7

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2139 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | elin, with whom he retained a long-standing relationship.\n |
| 2 | 2135 | `' a'` | latin_word | 35/35 | 2.2 | philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 3 | 1237 | `'.'` | punct_or_symbol | 35/35 | 3.9 | base of natural logarithms, e = 2.71828...), and found that |
| 4 | 2138 | `' relationship'` | latin_word | 35/35 | 3.9 | egelin, with whom he retained a long-standing relationship.\n |
| 5 | 2130 | `','` | punct_or_symbol | 35/35 | 5.1 | befriended noted political philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 6 | 2133 | `' he'` | latin_word | 35/35 | 6.1 | noted political philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 7 | 2137 | `'-standing'` | other_text | 35/35 | 7.1 | Voegelin, with whom he retained a long-standing relationship.\n |
| 8 | 2109 | `'.'` | punct_or_symbol | 35/35 | 8.8 | Hayek's own, more general, private seminar. It was during this time that he also encountered and |
| 9 | 2132 | `' whom'` | latin_word | 35/35 | 9.9 | ended noted political philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 10 | 2131 | `' with'` | latin_word | 35/35 | 10.2 | riended noted political philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 11 | 2006 | `' Hay'` | latin_word | 35/35 | 10.4 | of which are relatively well known.\nDocument: Friedrich Hayek: Initially sympathetic to Wieser's democratic |
| 12 | 2134 | `' retained'` | latin_word | 35/35 | 10.3 | political philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 13 | 2136 | `' long'` | latin_word | 35/35 | 13.3 | Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 14 | 1373 | `'ˈ'` | other_text | 35/35 | 15.8 | Feynman: Richard Phillips Feynman (/ˈfaɪnmən/; May 1 |
| 15 | 2119 | `' and'` | latin_word | 35/35 | 17.6 | . It was during this time that he also encountered and befriended noted political philosopher Eric Voegelin |
| 16 | 2125 | `' philosopher'` | latin_word | 35/35 | 18.3 | that he also encountered and befriended noted political philosopher Eric Voegelin, with whom he retained a |
| 17 | 2129 | `'elin'` | latin_word | 35/35 | 19.3 | and befriended noted political philosopher Eric Voegelin, with whom he retained a long-standing relationship.\n |
| 18 | 2127 | `' Vo'` | latin_word | 35/35 | 19.8 | also encountered and befriended noted political philosopher Eric Voegelin, with whom he retained a long-standing |
| 19 | 2124 | `' political'` | latin_word | 35/35 | 20.6 | time that he also encountered and befriended noted political philosopher Eric Voegelin, with whom he retained |
| 20 | 1818 | `'2'` | number | 35/35 | 23.3 | the US Federal Reserve.\nDocument: Josef Mach (25 February 1909 in Prost |
| 21 | 2110 | `' It'` | latin_word | 35/35 | 22.3 | ek's own, more general, private seminar. It was during this time that he also encountered and bef |
| 22 | 2123 | `' noted'` | latin_word | 35/35 | 23.1 | this time that he also encountered and befriended noted political philosopher Eric Voegelin, with whom he |
| 23 | 2122 | `'ended'` | latin_word | 35/35 | 24.8 | during this time that he also encountered and befriended noted political philosopher Eric Voegelin, with whom |
| 24 | 2118 | `' encountered'` | latin_word | 35/35 | 23.9 | seminar. It was during this time that he also encountered and befriended noted political philosopher Eric Voeg |
| 25 | 2126 | `' Eric'` | latin_word | 35/35 | 24.8 | he also encountered and befriended noted political philosopher Eric Voegelin, with whom he retained a long |
| 26 | 2115 | `' that'` | latin_word | 35/35 | 26.1 | general, private seminar. It was during this time that he also encountered and befriended noted political philosopher |
| 27 | 2116 | `' he'` | latin_word | 35/35 | 27.0 | , private seminar. It was during this time that he also encountered and befriended noted political philosopher Eric |
| 28 | 2113 | `' this'` | latin_word | 35/35 | 28.7 | , more general, private seminar. It was during this time that he also encountered and befriended noted |
| 29 | 2128 | `'eg'` | latin_word | 35/35 | 30.1 | encountered and befriended noted political philosopher Eric Voegelin, with whom he retained a long-standing relationship |
| 30 | 2010 | `' sympathetic'` | latin_word | 35/35 | 33.0 | well known.\nDocument: Friedrich Hayek: Initially sympathetic to Wieser's democratic socialism, Hayek |
| 31 | 2103 | `','` | punct_or_symbol | 35/35 | 32.1 | , who were also participating in Hayek's own, more general, private seminar. It was during this |
| 32 | 2111 | `' was'` | latin_word | 35/35 | 33.4 | 's own, more general, private seminar. It was during this time that he also encountered and befri |
| 33 | 2121 | `'ri'` | latin_word | 35/35 | 36.1 | was during this time that he also encountered and befriended noted political philosopher Eric Voegelin, with |
| 34 | 1059 | `'5'` | number | 35/35 | 46.2 | eventually published as The Sensory Order (1952). It located connective learning at the physical |
| 35 | 2114 | `' time'` | latin_word | 35/35 | 36.8 | more general, private seminar. It was during this time that he also encountered and befriended noted political |
| 36 | 2112 | `' during'` | latin_word | 35/35 | 36.5 | own, more general, private seminar. It was during this time that he also encountered and befriended |
| 37 | 2117 | `' also'` | latin_word | 35/35 | 38.9 | private seminar. It was during this time that he also encountered and befriended noted political philosopher Eric Vo |
| 38 | 2120 | `' bef'` | latin_word | 35/35 | 39.6 | It was during this time that he also encountered and befriended noted political philosopher Eric Voegelin, |
| 39 | 2046 | `'.'` | punct_or_symbol | 35/35 | 41.1 | enger after reading von Mises' book Socialism. It was sometime after reading Socialism that Hayek |
| 40 | 2106 | `','` | punct_or_symbol | 35/35 | 45.1 | also participating in Hayek's own, more general, private seminar. It was during this time that he |
| 41 | 2093 | `','` | punct_or_symbol | 35/35 | 44.9 | Felix Kaufmann, and Gottfried Haberler, who were also participating in Hayek's own, |
| 42 | 2078 | `','` | punct_or_symbol | 35/35 | 46.7 | of his university friends, including Fritz Machlup, Alfred Schutz, Felix Kaufmann, and Gott |
| 43 | 2065 | `','` | punct_or_symbol | 35/35 | 47.0 | Hayek began attending von Mises' private seminars, joining several of his university friends, including Fritz Mach |
| 44 | 2108 | `' seminar'` | latin_word | 35/35 | 48.5 | in Hayek's own, more general, private seminar. It was during this time that he also encountered |
| 45 | 1833 | `' '` | space | 35/35 | 64.6 | 909 in Prostějov – 7 July 1987 in Prague) |
| 46 | 2018 | `','` | punct_or_symbol | 35/35 | 53.3 | : Initially sympathetic to Wieser's democratic socialism, Hayek's economic thinking shifted away from socialism and |
| 47 | 2072 | `','` | punct_or_symbol | 35/35 | 51.9 | ' private seminars, joining several of his university friends, including Fritz Machlup, Alfred Schutz, |
| 48 | 2086 | `','` | punct_or_symbol | 35/35 | 52.7 | lup, Alfred Schutz, Felix Kaufmann, and Gottfried Haberler, who were also |
| 49 | 2082 | `','` | punct_or_symbol | 35/35 | 53.7 | , including Fritz Machlup, Alfred Schutz, Felix Kaufmann, and Gottfried Haberler |
| 50 | 2105 | `' general'` | latin_word | 35/35 | 54.2 | were also participating in Hayek's own, more general, private seminar. It was during this time that |
| 51 | 2101 | `"'s"` | other_text | 35/35 | 56.7 | erler, who were also participating in Hayek's own, more general, private seminar. It was |
| 52 | 2030 | `' the'` | latin_word | 35/35 | 59.3 | ek's economic thinking shifted away from socialism and toward the classical liberalism of Carl Menger after reading von M |
| 53 | 1603 | `' the'` | latin_word | 35/35 | 62.5 | vořák: He studied mathematics and physics at the Charles University in Prague, and after graduating he became |
| 54 | 1238 | `'7'` | number | 35/35 | 91.3 | of natural logarithms, e = 2.71828...), and found that the |
| 55 | 2002 | `'.\n'` | punct_or_symbol | 35/35 | 59.8 | in biology, some of which are relatively well known.\nDocument: Friedrich Hayek: Initially sympathetic to W |
| 56 | 2095 | `' were'` | latin_word | 35/35 | 60.0 | mann, and Gottfried Haberler, who were also participating in Hayek's own, more general |
| 57 | 1973 | `' the'` | latin_word | 35/35 | 63.4 | Edler von Hayek, taught natural sciences at the Imperial Realobergymnasium (secondary school |
| 58 | 903 | `'F'` | latin_word | 35/35 | 65.5 | renowned universities.\nDocument: The Fresnel number ("F"), named after the physicist Augustin-Jean Fres |
| 59 | 958 | `'9'` | number | 35/35 | 64.2 | doctorates in law and political science in 1921 and 1923 respectively; |
| 60 | 2102 | `' own'` | latin_word | 35/35 | 63.1 | ler, who were also participating in Hayek's own, more general, private seminar. It was during |
| 61 | 2094 | `' who'` | latin_word | 35/35 | 63.1 | Kaufmann, and Gottfried Haberler, who were also participating in Hayek's own, more |
| 62 | 604 | `'1'` | number | 35/35 | 76.0 | 7 (339 m / s, 1,221 km / h), may have |
| 63 | 1612 | `' he'` | latin_word | 35/35 | 77.4 | at the Charles University in Prague, and after graduating he became an assistant to professor Ernst Mach. After obtaining |
| 64 | 2107 | `' private'` | latin_word | 35/35 | 67.7 | participating in Hayek's own, more general, private seminar. It was during this time that he also |
| 65 | 2104 | `' more'` | latin_word | 35/35 | 68.2 | who were also participating in Hayek's own, more general, private seminar. It was during this time |
| 66 | 2087 | `' and'` | latin_word | 35/35 | 70.0 | up, Alfred Schutz, Felix Kaufmann, and Gottfried Haberler, who were also participating |
| 67 | 1862 | `' father'` | latin_word | 35/35 | 77.4 | film director.\nDocument: Friedrich Hayek: His father's career as a university professor influenced Friedrich's goals |
| 68 | 2023 | `' thinking'` | latin_word | 35/35 | 78.3 | ieser's democratic socialism, Hayek's economic thinking shifted away from socialism and toward the classical liberalism of |
| 69 | 2097 | `' participating'` | latin_word | 35/35 | 75.8 | and Gottfried Haberler, who were also participating in Hayek's own, more general, private |
| 70 | 2098 | `' in'` | latin_word | 35/35 | 77.4 | Gottfried Haberler, who were also participating in Hayek's own, more general, private seminar |
| 71 | 2037 | `' after'` | latin_word | 35/35 | 80.9 | socialism and toward the classical liberalism of Carl Menger after reading von Mises' book Socialism. It |
| 72 | 1958 | `' paternal'` | latin_word | 35/35 | 84.7 | was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav Edler von Hayek, |
| 73 | 1937 | `'.'` | punct_or_symbol | 35/35 | 82.4 | one of the founders of the Austrian School of Economics. Von Juraschek was a statistician and |
| 74 | 540 | `','` | punct_or_symbol | 35/35 | 102.8 | (361 m / s, 1,299 km / h, 80 |
| 75 | 2005 | `' Friedrich'` | latin_word | 35/35 | 104.4 | some of which are relatively well known.\nDocument: Friedrich Hayek: Initially sympathetic to Wieser's |
| 76 | 2096 | `' also'` | latin_word | 35/35 | 88.2 | , and Gottfried Haberler, who were also participating in Hayek's own, more general, |
| 77 | 2079 | `' Alfred'` | latin_word | 35/35 | 91.4 | his university friends, including Fritz Machlup, Alfred Schutz, Felix Kaufmann, and Gottfried |
| 78 | 2100 | `'ek'` | latin_word | 35/35 | 90.9 | Haberler, who were also participating in Hayek's own, more general, private seminar. It |
| 79 | 2071 | `' friends'` | latin_word | 35/35 | 91.3 | ises' private seminars, joining several of his university friends, including Fritz Machlup, Alfred Schutz |
| 80 | 1952 | `' the'` | latin_word | 35/35 | 101.1 | ek was a statistician and was later employed by the Austrian government. Friedrich's paternal grandfather, Gustav |

## Example 8

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2418 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | has been covered by hundreds of bands over the decades.\n |
| 2 | 2416 | `' the'` | latin_word | 35/35 | 2.1 | , and has been covered by hundreds of bands over the decades.\n |
| 3 | 2417 | `' decades'` | latin_word | 35/35 | 3.7 | and has been covered by hundreds of bands over the decades.\n |
| 4 | 1765 | `' re'` | latin_word | 35/35 | 5.4 | from December 1931 onward which was reissued on the Columbia label as well as the Vocal |
| 5 | 2406 | `','` | punct_or_symbol | 35/35 | 5.0 | Girl" has remained a staple on classic rock radio, and has been covered by hundreds of bands over the |
| 6 | 2400 | `' a'` | latin_word | 35/35 | 6.9 | song. "Brown Eyed Girl" has remained a staple on classic rock radio, and has been covered |
| 7 | 2415 | `' over'` | latin_word | 35/35 | 7.7 | radio, and has been covered by hundreds of bands over the decades.\n |
| 8 | 2391 | `'.'` | punct_or_symbol | 35/35 | 8.1 | and is considered to be Van Morrison's signature song. "Brown Eyed Girl" has remained a staple |
| 9 | 2414 | `' bands'` | latin_word | 35/35 | 9.6 | rock radio, and has been covered by hundreds of bands over the decades.\n |
| 10 | 2413 | `' of'` | latin_word | 35/35 | 9.9 | classic rock radio, and has been covered by hundreds of bands over the decades.\n |
| 11 | 1179 | `' '` | space | 35/35 | 14.9 | Concert for New York City for victims of the September 11 attacks in 2001, |
| 12 | 2407 | `' and'` | latin_word | 35/35 | 12.4 | " has remained a staple on classic rock radio, and has been covered by hundreds of bands over the decades |
| 13 | 2411 | `' by'` | latin_word | 35/35 | 12.3 | staple on classic rock radio, and has been covered by hundreds of bands over the decades.\n |
| 14 | 2412 | `' hundreds'` | latin_word | 35/35 | 15.2 | on classic rock radio, and has been covered by hundreds of bands over the decades.\n |
| 15 | 2410 | `' covered'` | latin_word | 35/35 | 15.9 | a staple on classic rock radio, and has been covered by hundreds of bands over the decades.\n |
| 16 | 2408 | `' has'` | latin_word | 35/35 | 16.1 | has remained a staple on classic rock radio, and has been covered by hundreds of bands over the decades.\n |
| 17 | 2397 | `'"'` | punct_or_symbol | 35/35 | 18.3 | Morrison's signature song. "Brown Eyed Girl" has remained a staple on classic rock radio, and |
| 18 | 2409 | `' been'` | latin_word | 35/35 | 19.4 | remained a staple on classic rock radio, and has been covered by hundreds of bands over the decades.\n |
| 19 | 2360 | `' the'` | latin_word | 35/35 | 22.4 | label, peaking at number 10 on the "Billboard" Hot 100. |
| 20 | 2373 | `' the'` | latin_word | 35/35 | 21.1 | board" Hot 100. It featured the Sweet Inspirations singing back-up vocals and is considered |
| 21 | 2402 | `' on'` | latin_word | 35/35 | 20.9 | "Brown Eyed Girl" has remained a staple on classic rock radio, and has been covered by hundreds |
| 22 | 2398 | `' has'` | latin_word | 35/35 | 22.5 | 's signature song. "Brown Eyed Girl" has remained a staple on classic rock radio, and has |
| 23 | 1255 | `' the'` | latin_word | 35/35 | 29.9 | , 2016, Joel performed at the BB&T Center in Sunrise, Florida, a city |
| 24 | 2405 | `' radio'` | latin_word | 35/35 | 25.2 | ed Girl" has remained a staple on classic rock radio, and has been covered by hundreds of bands over |
| 25 | 2390 | `' song'` | latin_word | 35/35 | 26.7 | vocals and is considered to be Van Morrison's signature song. "Brown Eyed Girl" has remained a |
| 26 | 2399 | `' remained'` | latin_word | 35/35 | 26.2 | signature song. "Brown Eyed Girl" has remained a staple on classic rock radio, and has been |
| 27 | 2404 | `' rock'` | latin_word | 35/35 | 27.9 | Eyed Girl" has remained a staple on classic rock radio, and has been covered by hundreds of bands |
| 28 | 2370 | `'.'` | punct_or_symbol | 35/35 | 27.9 | the "Billboard" Hot 100. It featured the Sweet Inspirations singing back-up vocals |
| 29 | 2392 | `' "'` | punct_or_symbol | 35/35 | 31.8 | is considered to be Van Morrison's signature song. "Brown Eyed Girl" has remained a staple on |
| 30 | 2403 | `' classic'` | latin_word | 35/35 | 31.6 | Brown Eyed Girl" has remained a staple on classic rock radio, and has been covered by hundreds of |
| 31 | 2401 | `' staple'` | latin_word | 35/35 | 31.9 | . "Brown Eyed Girl" has remained a staple on classic rock radio, and has been covered by |
| 32 | 2381 | `' and'` | latin_word | 35/35 | 32.5 | It featured the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's signature song. |
| 33 | 2396 | `' Girl'` | latin_word | 35/35 | 35.4 | Van Morrison's signature song. "Brown Eyed Girl" has remained a staple on classic rock radio, |
| 34 | 1786 | `' re'` | latin_word | 35/35 | 57.1 | ion label material from the same time period which was reissued on the Okeh label. Wallerstein |
| 35 | 2382 | `' is'` | latin_word | 35/35 | 38.5 | featured the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's signature song. " |
| 36 | 2393 | `'Brown'` | latin_word | 35/35 | 37.7 | considered to be Van Morrison's signature song. "Brown Eyed Girl" has remained a staple on classic |
| 37 | 2366 | `' '` | space | 35/35 | 42.3 | 10 on the "Billboard" Hot 100. It featured the Sweet Inspirations |
| 38 | 2385 | `' be'` | latin_word | 35/35 | 39.4 | Inspirations singing back-up vocals and is considered to be Van Morrison's signature song. "Brown Eyed |
| 39 | 2348 | `' the'` | latin_word | 35/35 | 39.9 | a single in June 1967 on the Bang label, peaking at number 10 |
| 40 | 2384 | `' to'` | latin_word | 35/35 | 45.3 | Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's signature song. "Brown Ey |
| 41 | 2383 | `' considered'` | latin_word | 35/35 | 42.8 | the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's signature song. "Brown |
| 42 | 2388 | `"'s"` | other_text | 35/35 | 44.1 | back-up vocals and is considered to be Van Morrison's signature song. "Brown Eyed Girl" has |
| 43 | 2389 | `' signature'` | latin_word | 35/35 | 45.9 | -up vocals and is considered to be Van Morrison's signature song. "Brown Eyed Girl" has remained |
| 44 | 2387 | `' Morrison'` | latin_word | 35/35 | 46.9 | singing back-up vocals and is considered to be Van Morrison's signature song. "Brown Eyed Girl" |
| 45 | 2395 | `'ed'` | latin_word | 35/35 | 47.2 | be Van Morrison's signature song. "Brown Eyed Girl" has remained a staple on classic rock radio |
| 46 | 2311 | `'.'` | punct_or_symbol | 35/35 | 48.0 | a song by Northern Irish singer and songwriter Van Morrison. Written by Morrison and recorded in March 19 |
| 47 | 2394 | `' Ey'` | latin_word | 35/35 | 48.6 | to be Van Morrison's signature song. "Brown Eyed Girl" has remained a staple on classic rock |
| 48 | 2351 | `','` | punct_or_symbol | 35/35 | 49.6 | June 1967 on the Bang label, peaking at number 10 on the " |
| 49 | 2386 | `' Van'` | latin_word | 35/35 | 51.7 | ations singing back-up vocals and is considered to be Van Morrison's signature song. "Brown Eyed Girl |
| 50 | 2362 | `'Bill'` | latin_word | 35/35 | 53.2 | peaking at number 10 on the "Billboard" Hot 100. It featured |
| 51 | 2299 | `'"'` | punct_or_symbol | 35/35 | 54.9 | ana Distribution.\nDocument: "Brown Eyed Girl" is a song by Northern Irish singer and songwriter Van |
| 52 | 2293 | `':'` | punct_or_symbol | 35/35 | 57.1 | . It is distributed by Fontana Distribution.\nDocument: "Brown Eyed Girl" is a song by |
| 53 | 2301 | `' a'` | latin_word | 35/35 | 56.8 | .\nDocument: "Brown Eyed Girl" is a song by Northern Irish singer and songwriter Van Morrison. |
| 54 | 2380 | `' vocals'` | latin_word | 35/35 | 55.9 | . It featured the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's signature song |
| 55 | 2364 | `'"'` | punct_or_symbol | 35/35 | 56.7 | at number 10 on the "Billboard" Hot 100. It featured the Sweet |
| 56 | 2338 | `' a'` | latin_word | 35/35 | 57.9 | and producer Bert Berns, it was released as a single in June 1967 on the |
| 57 | 2333 | `','` | punct_or_symbol | 35/35 | 57.3 | 7 for Bang Records owner and producer Bert Berns, it was released as a single in June 1 |
| 58 | 2356 | `' '` | space | 35/35 | 59.3 | 7 on the Bang label, peaking at number 10 on the "Billboard" Hot |
| 59 | 2372 | `' featured'` | latin_word | 35/35 | 58.7 | Billboard" Hot 100. It featured the Sweet Inspirations singing back-up vocals and is |
| 60 | 2284 | `' It'` | latin_word | 35/35 | 60.4 | " singles, CDs, and LP vinyl records. It is distributed by Fontana Distribution.\nDocument: " |
| 61 | 2291 | `'.\n'` | punct_or_symbol | 35/35 | 60.5 | vinyl records. It is distributed by Fontana Distribution.\nDocument: "Brown Eyed Girl" is a |
| 62 | 2378 | `' back'` | latin_word | 35/35 | 61.5 | 00. It featured the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's |
| 63 | 2068 | `'3'` | number | 35/35 | 68.1 | eme Goodall (1932 – 3 December 2014) was an Australian |
| 64 | 2379 | `'-up'` | other_text | 35/35 | 65.8 | 0. It featured the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison's signature |
| 65 | 1450 | `' the'` | latin_word | 35/35 | 70.6 | Flack in 1972, winning the Grammy Awards for Record and Song of the Year. |
| 66 | 1202 | `'0'` | number | 35/35 | 80.1 | television program "" for Hurricane Sandy victims in 2012 and during his set at "". Joel has |
| 67 | 2377 | `' singing'` | latin_word | 35/35 | 66.8 | 100. It featured the Sweet Inspirations singing back-up vocals and is considered to be Van Morrison |
| 68 | 2371 | `' It'` | latin_word | 35/35 | 67.3 | "Billboard" Hot 100. It featured the Sweet Inspirations singing back-up vocals and |
| 69 | 2099 | `','` | punct_or_symbol | 35/35 | 72.3 | figure in the early days of Jamaica's recording industry, constructing several of the Island's studios, co-f |
| 70 | 2376 | `'ations'` | latin_word | 35/35 | 73.5 | 100. It featured the Sweet Inspirations singing back-up vocals and is considered to be Van |
| 71 | 2367 | `'1'` | number | 35/35 | 79.5 | 10 on the "Billboard" Hot 100. It featured the Sweet Inspirations singing |
| 72 | 2363 | `'board'` | latin_word | 35/35 | 77.2 | aking at number 10 on the "Billboard" Hot 100. It featured the |
| 73 | 2160 | `'.'` | punct_or_symbol | 35/35 | 77.6 | today a European jazz label owned by Universal Music Group. The name is a phonetic spelling of "M |
| 74 | 2267 | `' '` | space | 35/35 | 117.7 | oto for their first EP. It has released over 40 records, 7" singles, CDs |
| 75 | 2234 | `' the'` | latin_word | 35/35 | 82.9 | Jawbox. It was first founded by members of the band Edsel to release their first single, " |
| 76 | 2283 | `'.'` | punct_or_symbol | 35/35 | 79.6 | 7" singles, CDs, and LP vinyl records. It is distributed by Fontana Distribution.\nDocument: |
| 77 | 2226 | `'.'` | punct_or_symbol | 35/35 | 80.0 | letta, both formerly of the band Jawbox. It was first founded by members of the band Ed |
| 78 | 2179 | `'.\n'` | punct_or_symbol | 35/35 | 81.8 | "MRC", the initials for Mercury Record Company.\nDocument: DeSoto Records is an American record |
| 79 | 2359 | `' on'` | latin_word | 35/35 | 87.1 | Bang label, peaking at number 10 on the "Billboard" Hot 100 |
| 80 | 2365 | `' Hot'` | latin_word | 35/35 | 84.7 | number 10 on the "Billboard" Hot 100. It featured the Sweet Inspir |

## Example 9

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 3106 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | French force sent overseas before the age of Louis XIV.\n |
| 2 | 2541 | `'1'` | number | 35/35 | 2.2 | coastal Crusader states.On 2 September 1192, following his defeat at Jaffa |
| 3 | 3105 | `' XIV'` | latin_word | 35/35 | 3.2 | largest French force sent overseas before the age of Louis XIV.\n |
| 4 | 3101 | `' the'` | latin_word | 35/35 | 3.7 | berian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 5 | 3104 | `' Louis'` | latin_word | 35/35 | 5.4 | —the largest French force sent overseas before the age of Louis XIV.\n |
| 6 | 3094 | `'—the'` | other_text | 35/35 | 6.7 | the islands from incorporation into the Iberian Union—the largest French force sent overseas before the age of Louis |
| 7 | 3089 | `' the'` | latin_word | 35/35 | 7.3 | Crato and to defend the islands from incorporation into the Iberian Union—the largest French force sent overseas |
| 8 | 3099 | `' overseas'` | latin_word | 35/35 | 9.0 | the Iberian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 9 | 3084 | `' the'` | latin_word | 35/35 | 10.6 | ónio, Prior of Crato and to defend the islands from incorporation into the Iberian Union—the |
| 10 | 3102 | `' age'` | latin_word | 35/35 | 10.9 | ian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 11 | 3103 | `' of'` | latin_word | 35/35 | 13.1 | Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 12 | 3100 | `' before'` | latin_word | 35/35 | 13.9 | Iberian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 13 | 3076 | `','` | punct_or_symbol | 35/35 | 15.7 | of the Azores under pretender António, Prior of Crato and to defend the islands from |
| 14 | 1815 | `' help'` | latin_word | 35/35 | 22.8 | ; Henry II was furious and ordered John, with help from Geoffrey, to march south and retake the |
| 15 | 3067 | `' the'` | latin_word | 35/35 | 17.1 | Portuguese and Castilian ships, to preserve control of the Azores under pretender António, Prior |
| 16 | 3098 | `' sent'` | latin_word | 35/35 | 16.4 | into the Iberian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 17 | 2860 | `'6'` | number | 35/35 | 18.8 | , resulting in the lifting of the siege in 678. The returning Muslim fleet suffered further losses |
| 18 | 3095 | `' largest'` | latin_word | 35/35 | 19.1 | islands from incorporation into the Iberian Union—the largest French force sent overseas before the age of Louis XIV |
| 19 | 2641 | `'1'` | number | 35/35 | 28.3 | departed the Holy Land on 9 October 1192.\nDocument: Arab–Byzantine |
| 20 | 3096 | `' French'` | latin_word | 35/35 | 20.0 | from incorporation into the Iberian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 21 | 3097 | `' force'` | latin_word | 35/35 | 22.2 | incorporation into the Iberian Union—the largest French force sent overseas before the age of Louis XIV.\n |
| 22 | 3081 | `' and'` | latin_word | 35/35 | 24.2 | pretender António, Prior of Crato and to defend the islands from incorporation into the Iber |
| 23 | 3093 | `' Union'` | latin_word | 35/35 | 25.0 | defend the islands from incorporation into the Iberian Union—the largest French force sent overseas before the age of |
| 24 | 3062 | `','` | punct_or_symbol | 35/35 | 26.1 | naval force made up of Portuguese and Castilian ships, to preserve control of the Azores under pretender |
| 25 | 3047 | `'),'` | punct_or_symbol | 35/35 | 27.7 | (an Anglo-French fleet with Portuguese forces included), sailed against a Spanish naval force made up of Portuguese |
| 26 | 3087 | `' incorporation'` | latin_word | 35/35 | 29.8 | Prior of Crato and to defend the islands from incorporation into the Iberian Union—the largest French force |
| 27 | 3090 | `' I'` | latin_word | 35/35 | 31.9 | ato and to defend the islands from incorporation into the Iberian Union—the largest French force sent overseas before |
| 28 | 1863 | `' the'` | latin_word | 35/35 | 34.5 | stalemate and a tense family reconciliation in England at the end of 1184.\nDocument: |
| 29 | 3085 | `' islands'` | latin_word | 35/35 | 32.9 | io, Prior of Crato and to defend the islands from incorporation into the Iberian Union—the largest |
| 30 | 3092 | `'ian'` | latin_word | 35/35 | 35.0 | to defend the islands from incorporation into the Iberian Union—the largest French force sent overseas before the age |
| 31 | 2591 | `' issue'` | latin_word | 35/35 | 42.7 | visit the city. Ascalon was a contentious issue as it threatened communication between Saladin's dominions |
| 32 | 3088 | `' into'` | latin_word | 35/35 | 36.4 | of Crato and to defend the islands from incorporation into the Iberian Union—the largest French force sent |
| 33 | 3086 | `' from'` | latin_word | 35/35 | 37.1 | , Prior of Crato and to defend the islands from incorporation into the Iberian Union—the largest French |
| 34 | 329 | `' the'` | latin_word | 35/35 | 43.4 | champ paid £3,000 for the office of Chancellor, and was soon named to the |
| 35 | 2536 | `' '` | space | 35/35 | 39.9 | strengthened the position of the coastal Crusader states.On 2 September 1192, following his |
| 36 | 2977 | `' "'` | punct_or_symbol | 35/35 | 47.7 | , "Battle of São Miguel" or specifically the "Battle of Vila Franca do Campo" took |
| 37 | 3091 | `'ber'` | latin_word | 35/35 | 37.4 | and to defend the islands from incorporation into the Iberian Union—the largest French force sent overseas before the |
| 38 | 2976 | `' the'` | latin_word | 35/35 | 45.6 | ada, "Battle of São Miguel" or specifically the "Battle of Vila Franca do Campo" |
| 39 | 3016 | `' the'` | latin_word | 35/35 | 40.7 | of São Miguel in the Portuguese archipelago of the Azores, during the War of the Portuguese Success |
| 40 | 3083 | `' defend'` | latin_word | 35/35 | 42.1 | António, Prior of Crato and to defend the islands from incorporation into the Iberian Union |
| 41 | 3028 | `'.'` | punct_or_symbol | 35/35 | 42.4 | ores, during the War of the Portuguese Succession. A combined corsair expedition, mainly French (an |
| 42 | 3058 | `' and'` | latin_word | 35/35 | 43.1 | sailed against a Spanish naval force made up of Portuguese and Castilian ships, to preserve control of the Az |
| 43 | 2745 | `'asts'` | latin_word | 35/35 | 47.0 | icus, from there they raided the Byzantine coasts almost at will. Finally in 676 |
| 44 | 3050 | `' a'` | latin_word | 35/35 | 44.5 | -French fleet with Portuguese forces included), sailed against a Spanish naval force made up of Portuguese and Castilian |
| 45 | 3082 | `' to'` | latin_word | 35/35 | 47.0 | ender António, Prior of Crato and to defend the islands from incorporation into the Iberian |
| 46 | 3045 | `' forces'` | latin_word | 35/35 | 53.1 | mainly French (an Anglo-French fleet with Portuguese forces included), sailed against a Spanish naval force made up |
| 47 | 3010 | `' the'` | latin_word | 35/35 | 50.7 | off the coast of the island of São Miguel in the Portuguese archipelago of the Azores, during |
| 48 | 3034 | `','` | punct_or_symbol | 35/35 | 50.5 | the Portuguese Succession. A combined corsair expedition, mainly French (an Anglo-French fleet with Portuguese |
| 49 | 3077 | `' Prior'` | latin_word | 35/35 | 50.2 | the Azores under pretender António, Prior of Crato and to defend the islands from incorporation |
| 50 | 3078 | `' of'` | latin_word | 35/35 | 52.4 | Azores under pretender António, Prior of Crato and to defend the islands from incorporation into |
| 51 | 3029 | `' A'` | latin_word | 35/35 | 53.7 | , during the War of the Portuguese Succession. A combined corsair expedition, mainly French (an Anglo |
| 52 | 2796 | `')'` | punct_or_symbol | 35/35 | 60.1 | r. 661–685) however used a devastating new weapon that came to be |
| 53 | 2481 | `'7'` | number | 35/35 | 63.7 | but his men were lightly armoured and lost 700 men killed due to the missiles of the |
| 54 | 3054 | `' made'` | latin_word | 35/35 | 65.1 | Portuguese forces included), sailed against a Spanish naval force made up of Portuguese and Castilian ships, to preserve |
| 55 | 3021 | `' the'` | latin_word | 35/35 | 61.7 | Portuguese archipelago of the Azores, during the War of the Portuguese Succession. A combined cors |
| 56 | 3080 | `'ato'` | latin_word | 35/35 | 60.3 | under pretender António, Prior of Crato and to defend the islands from incorporation into the I |
| 57 | 3070 | `' under'` | latin_word | 35/35 | 60.6 | ilian ships, to preserve control of the Azores under pretender António, Prior of Crato |
| 58 | 3037 | `' ('` | punct_or_symbol | 35/35 | 61.4 | ion. A combined corsair expedition, mainly French (an Anglo-French fleet with Portuguese forces included), |
| 59 | 3060 | `'ilian'` | latin_word | 35/35 | 69.9 | a Spanish naval force made up of Portuguese and Castilian ships, to preserve control of the Azores under |
| 60 | 2955 | `'.\n'` | punct_or_symbol | 35/35 | 61.9 | halted the Islamic expansion into Europe for almost thirty years.\nDocument: The naval Battle of Ponta Delg |
| 61 | 3079 | `' Cr'` | latin_word | 35/35 | 63.9 | ores under pretender António, Prior of Crato and to defend the islands from incorporation into the |
| 62 | 3074 | `'ón'` | other_text | 35/35 | 67.5 | preserve control of the Azores under pretender António, Prior of Crato and to defend the |
| 63 | 2999 | `','` | punct_or_symbol | 35/35 | 67.1 | on 26 July 1582, off the coast of the island of São Miguel in |
| 64 | 3019 | `','` | punct_or_symbol | 35/35 | 67.1 | in the Portuguese archipelago of the Azores, during the War of the Portuguese Succession. A |
| 65 | 2720 | `' re'` | latin_word | 35/35 | 94.5 | winter. Four years later, a massive Muslim fleet reappeared in the Marmara and re-established a |
| 66 | 3072 | `'ender'` | latin_word | 35/35 | 68.0 | , to preserve control of the Azores under pretender António, Prior of Crato and to |
| 67 | 3004 | `' the'` | latin_word | 35/35 | 70.0 | 1582, off the coast of the island of São Miguel in the Portuguese archipelago |
| 68 | 3057 | `' Portuguese'` | latin_word | 35/35 | 74.1 | ), sailed against a Spanish naval force made up of Portuguese and Castilian ships, to preserve control of the |
| 69 | 3069 | `'ores'` | latin_word | 35/35 | 76.4 | Castilian ships, to preserve control of the Azores under pretender António, Prior of Cr |
| 70 | 2994 | `' '` | space | 35/35 | 87.0 | do Campo" took place on 26 July 1582, off the coast of the |
| 71 | 3001 | `' the'` | latin_word | 35/35 | 77.5 | 26 July 1582, off the coast of the island of São Miguel in the Portuguese |
| 72 | 3073 | `' Ant'` | latin_word | 35/35 | 76.5 | to preserve control of the Azores under pretender António, Prior of Crato and to defend |
| 73 | 3075 | `'io'` | latin_word | 35/35 | 77.6 | control of the Azores under pretender António, Prior of Crato and to defend the islands |
| 74 | 3056 | `' of'` | latin_word | 35/35 | 80.0 | included), sailed against a Spanish naval force made up of Portuguese and Castilian ships, to preserve control of |
| 75 | 3066 | `' of'` | latin_word | 35/35 | 83.9 | of Portuguese and Castilian ships, to preserve control of the Azores under pretender António, |
| 76 | 3063 | `' to'` | latin_word | 35/35 | 79.3 | force made up of Portuguese and Castilian ships, to preserve control of the Azores under pretender Ant |
| 77 | 3068 | `' Az'` | latin_word | 35/35 | 81.0 | and Castilian ships, to preserve control of the Azores under pretender António, Prior of |
| 78 | 2352 | `'2'` | number | 35/35 | 90.9 | Richard and a small force of little more than 2,000 men went to Jaffa |
| 79 | 3064 | `' preserve'` | latin_word | 35/35 | 83.6 | made up of Portuguese and Castilian ships, to preserve control of the Azores under pretender Antón |
| 80 | 3071 | `' pret'` | latin_word | 35/35 | 84.1 | ships, to preserve control of the Azores under pretender António, Prior of Crato and |

## Example 10

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2347 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | Peter and Stowe both live in Nashville, Tennessee.\n |
| 2 | 2345 | `','` | punct_or_symbol | 35/35 | 2.1 | others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 3 | 2336 | `'.'` | punct_or_symbol | 35/35 | 3.4 | experience as an abused child in order to help others. Peter and Stowe both live in Nashville, Tennessee |
| 4 | 2346 | `' Tennessee'` | latin_word | 35/35 | 4.8 | . Peter and Stowe both live in Nashville, Tennessee.\n |
| 5 | 2304 | `' the'` | latin_word | 35/35 | 5.6 | 700 Club. Peter co-wrote the book "Journey of Light" with Stowe |
| 6 | 2343 | `' in'` | latin_word | 35/35 | 6.5 | to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 7 | 2338 | `' and'` | latin_word | 35/35 | 8.2 | an abused child in order to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 8 | 2251 | `' a'` | latin_word | 35/35 | 9.2 | an American filmmaker and screenwriter. He has received a New York City Film Festival award for his television film |
| 9 | 2344 | `' Nashville'` | latin_word | 35/35 | 8.5 | help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 10 | 2342 | `' live'` | latin_word | 35/35 | 11.3 | order to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 11 | 2339 | `' St'` | latin_word | 35/35 | 13.0 | abused child in order to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 12 | 2337 | `' Peter'` | latin_word | 35/35 | 14.0 | as an abused child in order to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 13 | 2173 | `' the'` | latin_word | 35/35 | 19.2 | the years following his studies of the Italian Renaissance at the University of Florence.\nDocument: Rex Wockner |
| 14 | 2341 | `' both'` | latin_word | 35/35 | 15.2 | in order to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 15 | 2328 | `' an'` | latin_word | 35/35 | 16.2 | ockey and its focus is to implement her experience as an abused child in order to help others. Peter and |
| 16 | 2319 | `' and'` | latin_word | 35/35 | 18.0 | of Light" with Stowe D. Shockey and its focus is to implement her experience as an abused |
| 17 | 2333 | `' to'` | latin_word | 35/35 | 25.2 | to implement her experience as an abused child in order to help others. Peter and Stowe both live in |
| 18 | 2218 | `' has'` | latin_word | 35/35 | 26.5 | icals since 1985. His work has appeared in more than 325 gay publications |
| 19 | 2340 | `'owe'` | latin_word | 35/35 | 22.0 | child in order to help others. Peter and Stowe both live in Nashville, Tennessee.\n |
| 20 | 2299 | `'.'` | punct_or_symbol | 35/35 | 22.5 | such as Oprah and the 700 Club. Peter co-wrote the book "Journey of |
| 21 | 2329 | `' abused'` | latin_word | 35/35 | 22.5 | and its focus is to implement her experience as an abused child in order to help others. Peter and St |
| 22 | 2335 | `' others'` | latin_word | 35/35 | 24.7 | her experience as an abused child in order to help others. Peter and Stowe both live in Nashville, |
| 23 | 2331 | `' in'` | latin_word | 35/35 | 24.0 | focus is to implement her experience as an abused child in order to help others. Peter and Stowe both |
| 24 | 2326 | `' experience'` | latin_word | 35/35 | 24.8 | . Shockey and its focus is to implement her experience as an abused child in order to help others. |
| 25 | 2330 | `' child'` | latin_word | 35/35 | 26.6 | its focus is to implement her experience as an abused child in order to help others. Peter and Stowe |
| 26 | 2323 | `' to'` | latin_word | 35/35 | 29.4 | Stowe D. Shockey and its focus is to implement her experience as an abused child in order to |
| 27 | 2324 | `' implement'` | latin_word | 35/35 | 29.5 | owe D. Shockey and its focus is to implement her experience as an abused child in order to help |
| 28 | 2301 | `' co'` | latin_word | 35/35 | 37.7 | Oprah and the 700 Club. Peter co-wrote the book "Journey of Light" |
| 29 | 2325 | `' her'` | latin_word | 35/35 | 30.3 | D. Shockey and its focus is to implement her experience as an abused child in order to help others |
| 30 | 2004 | `'.'` | punct_or_symbol | 35/35 | 33.4 | resolution into law on December 18 as Pub. L. 107 -- 89 |
| 31 | 2332 | `' order'` | latin_word | 35/35 | 31.0 | is to implement her experience as an abused child in order to help others. Peter and Stowe both live |
| 32 | 2322 | `' is'` | latin_word | 35/35 | 32.4 | with Stowe D. Shockey and its focus is to implement her experience as an abused child in order |
| 33 | 2274 | `' addition'` | latin_word | 35/35 | 32.2 | After Life" (1992). In addition, both he and his works have appeared on talk |
| 34 | 2188 | `'9'` | number | 35/35 | 55.8 | Document: Rex Wockner (born 1957) is an American freelance journalist who has |
| 35 | 2327 | `' as'` | latin_word | 35/35 | 36.6 | Shockey and its focus is to implement her experience as an abused child in order to help others. Peter |
| 36 | 2334 | `' help'` | latin_word | 35/35 | 36.7 | implement her experience as an abused child in order to help others. Peter and Stowe both live in Nashville |
| 37 | 2316 | `'.'` | punct_or_symbol | 35/35 | 37.3 | "Journey of Light" with Stowe D. Shockey and its focus is to implement her experience |
| 38 | 2311 | `'"'` | punct_or_symbol | 35/35 | 37.3 | co-wrote the book "Journey of Light" with Stowe D. Shockey and its focus |
| 39 | 474 | `'2'` | number | 35/35 | 52.0 | and the James Tait Black Memorial Prize in 2002.\nDocument: George Saunders (born |
| 40 | 2312 | `' with'` | latin_word | 35/35 | 39.9 | -wrote the book "Journey of Light" with Stowe D. Shockey and its focus is |
| 41 | 2321 | `' focus'` | latin_word | 35/35 | 40.6 | " with Stowe D. Shockey and its focus is to implement her experience as an abused child in |
| 42 | 2320 | `' its'` | latin_word | 35/35 | 43.1 | Light" with Stowe D. Shockey and its focus is to implement her experience as an abused child |
| 43 | 1128 | `' '` | space | 35/35 | 48.5 | - 09 - 28) September 28, 1969 (age |
| 44 | 2217 | `' work'` | latin_word | 35/35 | 51.9 | periodicals since 1985. His work has appeared in more than 325 gay |
| 45 | 2293 | `' the'` | latin_word | 35/35 | 46.8 | on talk shows and television shows such as Oprah and the 700 Club. Peter co-wrote |
| 46 | 2291 | `' Oprah'` | latin_word | 35/35 | 48.0 | have appeared on talk shows and television shows such as Oprah and the 700 Club. Peter co |
| 47 | 2215 | `'.'` | punct_or_symbol | 35/35 | 51.9 | and mainstream periodicals since 1985. His work has appeared in more than 32 |
| 48 | 2306 | `' "'` | punct_or_symbol | 35/35 | 49.6 | 00 Club. Peter co-wrote the book "Journey of Light" with Stowe D. |
| 49 | 2272 | `').'` | punct_or_symbol | 35/35 | 49.2 | "Life After Life" (1992). In addition, both he and his works have appeared |
| 50 | 2247 | `'.'` | punct_or_symbol | 35/35 | 50.6 | Peter Shockey is an American filmmaker and screenwriter. He has received a New York City Film Festival award |
| 51 | 2318 | `'ockey'` | latin_word | 35/35 | 51.8 | ourney of Light" with Stowe D. Shockey and its focus is to implement her experience as an |
| 52 | 2202 | `' the'` | latin_word | 35/35 | 57.6 | is an American freelance journalist who has reported news for the gay press and mainstream periodicals since 19 |
| 53 | 2163 | `' the'` | latin_word | 35/35 | 65.9 | novel of Leonardo da Vinci. It was conceived in the years following his studies of the Italian Renaissance at the |
| 54 | 2286 | `' and'` | latin_word | 35/35 | 57.3 | both he and his works have appeared on talk shows and television shows such as Oprah and the 70 |
| 55 | 2234 | `'.\n'` | punct_or_symbol | 35/35 | 57.6 | 325 gay publications in 38 countries.\nDocument: Peter Shockey is an American filmmaker and |
| 56 | 2292 | `' and'` | latin_word | 35/35 | 58.0 | appeared on talk shows and television shows such as Oprah and the 700 Club. Peter co-w |
| 57 | 2305 | `' book'` | latin_word | 35/35 | 61.6 | 700 Club. Peter co-wrote the book "Journey of Light" with Stowe D |
| 58 | 2315 | `' D'` | latin_word | 35/35 | 64.8 | book "Journey of Light" with Stowe D. Shockey and its focus is to implement her |
| 59 | 2317 | `' Sh'` | latin_word | 35/35 | 65.2 | Journey of Light" with Stowe D. Shockey and its focus is to implement her experience as |
| 60 | 2275 | `','` | punct_or_symbol | 35/35 | 64.2 | Life" (1992). In addition, both he and his works have appeared on talk shows |
| 61 | 2310 | `' Light'` | latin_word | 35/35 | 66.5 | Peter co-wrote the book "Journey of Light" with Stowe D. Shockey and its |
| 62 | 2314 | `'owe'` | latin_word | 35/35 | 66.4 | the book "Journey of Light" with Stowe D. Shockey and its focus is to implement |
| 63 | 2244 | `' and'` | latin_word | 35/35 | 68.9 | .\nDocument: Peter Shockey is an American filmmaker and screenwriter. He has received a New York City |
| 64 | 2241 | `' an'` | latin_word | 35/35 | 68.5 | 38 countries.\nDocument: Peter Shockey is an American filmmaker and screenwriter. He has received a |
| 65 | 2313 | `' St'` | latin_word | 35/35 | 69.8 | rote the book "Journey of Light" with Stowe D. Shockey and its focus is to |
| 66 | 2308 | `'ourney'` | latin_word | 35/35 | 73.3 | Club. Peter co-wrote the book "Journey of Light" with Stowe D. Shockey |
| 67 | 2302 | `'-w'` | other_text | 35/35 | 80.3 | and the 700 Club. Peter co-wrote the book "Journey of Light" with |
| 68 | 2309 | `' of'` | latin_word | 35/35 | 74.8 | . Peter co-wrote the book "Journey of Light" with Stowe D. Shockey and |
| 69 | 2259 | `' his'` | latin_word | 35/35 | 86.5 | has received a New York City Film Festival award for his television film "Life After Life" (19 |
| 70 | 2236 | `':'` | punct_or_symbol | 35/35 | 75.3 | 5 gay publications in 38 countries.\nDocument: Peter Shockey is an American filmmaker and screenwriter |
| 71 | 2303 | `'rote'` | latin_word | 35/35 | 76.5 | the 700 Club. Peter co-wrote the book "Journey of Light" with St |
| 72 | 2300 | `' Peter'` | latin_word | 35/35 | 78.6 | as Oprah and the 700 Club. Peter co-wrote the book "Journey of Light |
| 73 | 2102 | `'9'` | number | 35/35 | 112.3 | week"s Mediterranean bureau chief in Rome from 1957 to 1969. He |
| 74 | 2240 | `' is'` | latin_word | 35/35 | 78.8 | 38 countries.\nDocument: Peter Shockey is an American filmmaker and screenwriter. He has received |
| 75 | 2223 | `' '` | space | 35/35 | 113.5 | 85. His work has appeared in more than 325 gay publications in 38 countries |
| 76 | 2294 | `' '` | space | 35/35 | 80.7 | talk shows and television shows such as Oprah and the 700 Club. Peter co-wrote the |
| 77 | 2289 | `' such'` | latin_word | 35/35 | 88.4 | his works have appeared on talk shows and television shows such as Oprah and the 700 Club. |
| 78 | 2122 | `' the'` | latin_word | 35/35 | 88.1 | He also worked for Edward R. Murrow at the Rome bureau of CBS, and covered the Vatican for |
| 79 | 1432 | `' levels'` | latin_word | 35/35 | 111.5 | with gaining her students' trust on personal and academic levels, but she must do so with very little support |
| 80 | 2298 | `' Club'` | latin_word | 35/35 | 92.8 | shows such as Oprah and the 700 Club. Peter co-wrote the book "Journey |

## Example 11

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2330 | `'.\n'` | punct_or_symbol | 35/35 | 1.1 | of Charm", a variety show of cabaret acts.\n |
| 2 | 2323 | `' a'` | latin_word | 35/35 | 1.9 | of "Sxip's Hour of Charm", a variety show of cabaret acts.\n |
| 3 | 2322 | `'",'` | punct_or_symbol | 35/35 | 3.0 | producer of "Sxip's Hour of Charm", a variety show of cabaret acts.\n |
| 4 | 2329 | `' acts'` | latin_word | 35/35 | 4.0 | Hour of Charm", a variety show of cabaret acts.\n |
| 5 | 2306 | `'.'` | punct_or_symbol | 35/35 | 5.7 | as well as the band Gentlemen & Assassins. He is the host and producer of "Sx |
| 6 | 2328 | `'aret'` | latin_word | 35/35 | 5.8 | 's Hour of Charm", a variety show of cabaret acts.\n |
| 7 | 2309 | `' the'` | latin_word | 35/35 | 8.4 | the band Gentlemen & Assassins. He is the host and producer of "Sxip's Hour |
| 8 | 2295 | `','` | punct_or_symbol | 35/35 | 9.4 | of the band Luminescent Orchestrii, as well as the band Gentlemen & Assassins |
| 9 | 2325 | `' show'` | latin_word | 35/35 | 8.9 | Sxip's Hour of Charm", a variety show of cabaret acts.\n |
| 10 | 2326 | `' of'` | latin_word | 35/35 | 10.4 | xip's Hour of Charm", a variety show of cabaret acts.\n |
| 11 | 2286 | `' the'` | latin_word | 35/35 | 11.6 | Daredevil Opera Company and is a founding member of the band Luminescent Orchestrii, as |
| 12 | 2324 | `' variety'` | latin_word | 35/35 | 12.7 | "Sxip's Hour of Charm", a variety show of cabaret acts.\n |
| 13 | 2311 | `' and'` | latin_word | 35/35 | 12.9 | Gentlemen & Assassins. He is the host and producer of "Sxip's Hour of Charm |
| 14 | 2327 | `' cab'` | latin_word | 35/35 | 13.3 | ip's Hour of Charm", a variety show of cabaret acts.\n |
| 15 | 2313 | `' of'` | latin_word | 35/35 | 16.8 | & Assassins. He is the host and producer of "Sxip's Hour of Charm", a |
| 16 | 2307 | `' He'` | latin_word | 35/35 | 18.3 | well as the band Gentlemen & Assassins. He is the host and producer of "Sxip |
| 17 | 2321 | `' Charm'` | latin_word | 35/35 | 18.9 | and producer of "Sxip's Hour of Charm", a variety show of cabaret acts.\n |
| 18 | 2318 | `"'s"` | other_text | 35/35 | 19.3 | is the host and producer of "Sxip's Hour of Charm", a variety show of cabaret |
| 19 | 2308 | `' is'` | latin_word | 35/35 | 19.0 | as the band Gentlemen & Assassins. He is the host and producer of "Sxip's |
| 20 | 2191 | `'S'` | latin_word | 35/35 | 22.8 | : Sxip Shirey: Gene "Sxip" Shirey (pronounced " |
| 21 | 2299 | `' the'` | latin_word | 35/35 | 20.4 | inescent Orchestrii, as well as the band Gentlemen & Assassins. He is the |
| 22 | 2319 | `' Hour'` | latin_word | 35/35 | 23.2 | the host and producer of "Sxip's Hour of Charm", a variety show of cabaret acts |
| 23 | 2192 | `'x'` | latin_word | 35/35 | 27.4 | Sxip Shirey: Gene "Sxip" Shirey (pronounced "skip |
| 24 | 2314 | `' "'` | punct_or_symbol | 35/35 | 24.7 | Assassins. He is the host and producer of "Sxip's Hour of Charm", a variety |
| 25 | 2312 | `' producer'` | latin_word | 35/35 | 25.4 | lemen & Assassins. He is the host and producer of "Sxip's Hour of Charm", |
| 26 | 2310 | `' host'` | latin_word | 35/35 | 26.7 | band Gentlemen & Assassins. He is the host and producer of "Sxip's Hour of |
| 27 | 2023 | `' the'` | latin_word | 35/35 | 28.8 | -rock styles. The former four groups were included on the Eno-produced No New York compilation, often considered |
| 28 | 2320 | `' of'` | latin_word | 35/35 | 29.7 | host and producer of "Sxip's Hour of Charm", a variety show of cabaret acts.\n |
| 29 | 2253 | `','` | punct_or_symbol | 35/35 | 32.1 | instruments. Shirey has released three solo albums, including "Sonic New York" in 2 |
| 30 | 2315 | `'S'` | latin_word | 35/35 | 30.5 | ins. He is the host and producer of "Sxip's Hour of Charm", a variety show |
| 31 | 2316 | `'x'` | latin_word | 35/35 | 36.6 | . He is the host and producer of "Sxip's Hour of Charm", a variety show of |
| 32 | 2272 | `' a'` | latin_word | 35/35 | 36.1 | 2010. Shirey is a member of The Daredevil Opera Company and is a |
| 33 | 2317 | `'ip'` | latin_word | 35/35 | 37.6 | He is the host and producer of "Sxip's Hour of Charm", a variety show of cab |
| 34 | 2280 | `' and'` | latin_word | 35/35 | 36.7 | y is a member of The Daredevil Opera Company and is a founding member of the band Luminescent |
| 35 | 2267 | `'.'` | punct_or_symbol | 35/35 | 37.1 | onic New York" in 2010. Shirey is a member of The Daredevil |
| 36 | 2282 | `' a'` | latin_word | 35/35 | 38.7 | a member of The Daredevil Opera Company and is a founding member of the band Luminescent Orchest |
| 37 | 2244 | `'.'` | punct_or_symbol | 35/35 | 42.4 | , traditional instruments, and computer and rare modified instruments. Shirey has released three solo albums, including |
| 38 | 2302 | `'lemen'` | latin_word | 35/35 | 42.9 | chestrii, as well as the band Gentlemen & Assassins. He is the host and producer |
| 39 | 2303 | `' &'` | punct_or_symbol | 35/35 | 42.9 | rii, as well as the band Gentlemen & Assassins. He is the host and producer of |
| 40 | 2305 | `'ins'` | latin_word | 35/35 | 46.1 | , as well as the band Gentlemen & Assassins. He is the host and producer of "S |
| 41 | 1651 | `' '` | space | 35/35 | 46.0 | 2006.\nDocument: Golgoth 13 were a French band, formed in |
| 42 | 2301 | `' Gent'` | latin_word | 35/35 | 45.3 | Orchestrii, as well as the band Gentlemen & Assassins. He is the host and |
| 43 | 2300 | `' band'` | latin_word | 35/35 | 48.4 | cent Orchestrii, as well as the band Gentlemen & Assassins. He is the host |
| 44 | 21 | `' re'` | latin_word | 35/35 | 56.3 | album by British rock band Art Brut. It was re-released in 2006 with bonus |
| 45 | 2304 | `' Assass'` | latin_word | 35/35 | 48.1 | i, as well as the band Gentlemen & Assassins. He is the host and producer of " |
| 46 | 2220 | `' based'` | latin_word | 35/35 | 65.0 | composer, performer, and story-teller. Currently based in New York City, he is known for working |
| 47 | 2296 | `' as'` | latin_word | 35/35 | 48.5 | the band Luminescent Orchestrii, as well as the band Gentlemen & Assassins. |
| 48 | 2297 | `' well'` | latin_word | 35/35 | 50.4 | band Luminescent Orchestrii, as well as the band Gentlemen & Assassins. He |
| 49 | 2218 | `'.'` | punct_or_symbol | 35/35 | 52.1 | -acoustic composer, performer, and story-teller. Currently based in New York City, he is known |
| 50 | 2271 | `' is'` | latin_word | 35/35 | 54.3 | in 2010. Shirey is a member of The Daredevil Opera Company and is |
| 51 | 2275 | `' The'` | latin_word | 35/35 | 54.0 | 10. Shirey is a member of The Daredevil Opera Company and is a founding member of |
| 52 | 2080 | `' the'` | latin_word | 35/35 | 55.8 | subsequent years.\nDocument: The Incredible Sound Machine is the fifth and final album by old school hip hop/e |
| 53 | 2287 | `' band'` | latin_word | 35/35 | 62.8 | devil Opera Company and is a founding member of the band Luminescent Orchestrii, as well |
| 54 | 2200 | `'ounced'` | latin_word | 35/35 | 62.5 | "Sxip" Shirey (pronounced "skip") is an American electric-acoustic composer |
| 55 | 2294 | `'i'` | latin_word | 35/35 | 59.6 | member of the band Luminescent Orchestrii, as well as the band Gentlemen & Assass |
| 56 | 2204 | `' is'` | latin_word | 35/35 | 74.1 | " Shirey (pronounced "skip") is an American electric-acoustic composer, performer, and |
| 57 | 2293 | `'ri'` | latin_word | 35/35 | 60.5 | founding member of the band Luminescent Orchestrii, as well as the band Gentlemen & |
| 58 | 2211 | `','` | punct_or_symbol | 35/35 | 60.9 | "skip") is an American electric-acoustic composer, performer, and story-teller. Currently based in |
| 59 | 2195 | `' Sh'` | latin_word | 35/35 | 69.6 | Shirey: Gene "Sxip" Shirey (pronounced "skip") is an |
| 60 | 2298 | `' as'` | latin_word | 35/35 | 61.4 | Luminescent Orchestrii, as well as the band Gentlemen & Assassins. He is |
| 61 | 2281 | `' is'` | latin_word | 35/35 | 65.8 | is a member of The Daredevil Opera Company and is a founding member of the band Luminescent Or |
| 62 | 2285 | `' of'` | latin_word | 35/35 | 67.0 | The Daredevil Opera Company and is a founding member of the band Luminescent Orchestrii, |
| 63 | 2283 | `' founding'` | latin_word | 35/35 | 68.5 | member of The Daredevil Opera Company and is a founding member of the band Luminescent Orchestri |
| 64 | 2292 | `'chest'` | latin_word | 35/35 | 68.4 | a founding member of the band Luminescent Orchestrii, as well as the band Gentlemen |
| 65 | 2234 | `','` | punct_or_symbol | 35/35 | 70.4 | City, he is known for working with found objects, traditional instruments, and computer and rare modified instruments. |
| 66 | 2072 | `'.\n'` | punct_or_symbol | 35/35 | 74.0 | also produce acclaimed and influential compilations in subsequent years.\nDocument: The Incredible Sound Machine is the fifth and |
| 67 | 2225 | `','` | punct_or_symbol | 35/35 | 73.7 | story-teller. Currently based in New York City, he is known for working with found objects, traditional |
| 68 | 2262 | `' '` | space | 35/35 | 73.8 | albums, including "Sonic New York" in 2010. Shirey is a |
| 69 | 2284 | `' member'` | latin_word | 35/35 | 74.9 | of The Daredevil Opera Company and is a founding member of the band Luminescent Orchestrii |
| 70 | 2274 | `' of'` | latin_word | 35/35 | 79.2 | 010. Shirey is a member of The Daredevil Opera Company and is a founding member |
| 71 | 2291 | `' Or'` | latin_word | 35/35 | 77.5 | is a founding member of the band Luminescent Orchestrii, as well as the band Gent |
| 72 | 2159 | `'9'` | number | 35/35 | 83.2 | on Mantronix's previous album, 1990's "This Should Move Ya"), and |
| 73 | 2260 | `'"'` | punct_or_symbol | 35/35 | 80.3 | three solo albums, including "Sonic New York" in 2010. Shirey |
| 74 | 2179 | `'.\n'` | punct_or_symbol | 35/35 | 80.2 | and founding member, DJ Kurtis Mantronik.\nDocument: Sxip Shirey: Gene |
| 75 | 2279 | `' Company'` | latin_word | 35/35 | 84.3 | irey is a member of The Daredevil Opera Company and is a founding member of the band Lumines |
| 76 | 2288 | `' Lum'` | latin_word | 35/35 | 83.8 | Opera Company and is a founding member of the band Luminescent Orchestrii, as well as |
| 77 | 2273 | `' member'` | latin_word | 35/35 | 85.1 | 2010. Shirey is a member of The Daredevil Opera Company and is a founding |
| 78 | 1580 | `' the'` | latin_word | 35/35 | 101.7 | Rockparty) - the biggest Swedish music festival at the time and an opportunity for unsigned bands to perform at |
| 79 | 1952 | `' trop'` | latin_word | 35/35 | 99.7 | a reaction against punk's recycling of traditionalist rock tropes and often reflected an abrasive, confrontational and |
| 80 | 2276 | `' Dare'` | latin_word | 35/35 | 89.2 | 0. Shirey is a member of The Daredevil Opera Company and is a founding member of the |

## Example 12

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2461 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | the basic PA-8000 processor core.\n |
| 2 | 2451 | `' the'` | latin_word | 35/35 | 2.1 | 00, described further below) are based on the basic PA-8000 processor core.\n |
| 3 | 2460 | `' core'` | latin_word | 35/35 | 3.3 | on the basic PA-8000 processor core.\n |
| 4 | 2446 | `' below'` | latin_word | 35/35 | 4.3 | to PA-8900, described further below) are based on the basic PA-80 |
| 5 | 2459 | `' processor'` | latin_word | 35/35 | 4.9 | based on the basic PA-8000 processor core.\n |
| 6 | 2447 | `')'` | punct_or_symbol | 35/35 | 6.2 | PA-8900, described further below) are based on the basic PA-800 |
| 7 | 2415 | `' the'` | latin_word | 35/35 | 9.1 | used exclusively by PRO members and was not sold on the merchant market. All follow-on PA-8x |
| 8 | 2448 | `' are'` | latin_word | 35/35 | 8.5 | -8900, described further below) are based on the basic PA-8000 |
| 9 | 2443 | `','` | punct_or_symbol | 35/35 | 9.5 | 200 to PA-8900, described further below) are based on the basic PA |
| 10 | 2450 | `' on'` | latin_word | 35/35 | 10.1 | 900, described further below) are based on the basic PA-8000 processor core |
| 11 | 2449 | `' based'` | latin_word | 35/35 | 11.3 | 8900, described further below) are based on the basic PA-8000 processor |
| 12 | 2453 | `' PA'` | latin_word | 35/35 | 13.4 | , described further below) are based on the basic PA-8000 processor core.\n |
| 13 | 2455 | `'8'` | number | 35/35 | 16.1 | further below) are based on the basic PA-8000 processor core.\n |
| 14 | 2458 | `'0'` | number | 35/35 | 15.5 | are based on the basic PA-8000 processor core.\n |
| 15 | 2452 | `' basic'` | latin_word | 35/35 | 16.5 | 0, described further below) are based on the basic PA-8000 processor core.\n |
| 16 | 2418 | `'.'` | punct_or_symbol | 35/35 | 17.2 | PRO members and was not sold on the merchant market. All follow-on PA-8x00 processors |
| 17 | 2444 | `' described'` | latin_word | 35/35 | 17.4 | 00 to PA-8900, described further below) are based on the basic PA- |
| 18 | 2457 | `'0'` | number | 35/35 | 19.5 | ) are based on the basic PA-8000 processor core.\n |
| 19 | 2456 | `'0'` | number | 35/35 | 20.2 | below) are based on the basic PA-8000 processor core.\n |
| 20 | 2454 | `'-'` | punct_or_symbol | 35/35 | 21.1 | described further below) are based on the basic PA-8000 processor core.\n |
| 21 | 2402 | `').'` | punct_or_symbol | 35/35 | 21.3 | to members of the Precision RISC Organization (PRO). It was used exclusively by PRO members and was not |
| 22 | 2001 | `'D'` | latin_word | 35/35 | 30.4 | 22%, as well as reduce the "Dell Hell" prominent on Internet search engines.\nDocument |
| 23 | 1452 | `','` | punct_or_symbol | 35/35 | 25.4 | the MacBook Air) compared to 500,000 total Ultrabooks, although there |
| 24 | 2395 | `' the'` | latin_word | 35/35 | 23.5 | 1995 when shipments began to members of the Precision RISC Organization (PRO). It was used |
| 25 | 2445 | `' further'` | latin_word | 35/35 | 25.9 | 0 to PA-8900, described further below) are based on the basic PA-8 |
| 26 | 2429 | `' ('` | punct_or_symbol | 35/35 | 26.4 | All follow-on PA-8x00 processors (PA-8200 to PA-8 |
| 27 | 1636 | `' '` | space | 35/35 | 29.6 | 2013, the MacBook Air took in 56 percent of all Ultrabook sales in |
| 28 | 2226 | `' '` | space | 35/35 | 42.7 | 2002.\nDocument: On February 8, 2011, the Mem |
| 29 | 2428 | `' processors'` | latin_word | 35/35 | 30.9 | . All follow-on PA-8x00 processors (PA-8200 to PA- |
| 30 | 2436 | `' to'` | latin_word | 35/35 | 31.9 | 00 processors (PA-8200 to PA-8900, described further below |
| 31 | 2410 | `' and'` | latin_word | 35/35 | 32.9 | (PRO). It was used exclusively by PRO members and was not sold on the merchant market. All follow |
| 32 | 2417 | `' market'` | latin_word | 35/35 | 48.3 | by PRO members and was not sold on the merchant market. All follow-on PA-8x00 |
| 33 | 1473 | `' the'` | latin_word | 35/35 | 68.8 | were dozens of Ultrabooks from various manufacturers on the market while Apple only offered 11-inch and |
| 34 | 2381 | `' '` | space | 35/35 | 50.1 | The PA-8000 was introduced on 2 November 1995 when shipments began |
| 35 | 2370 | `'.'` | punct_or_symbol | 35/35 | 38.7 | circuitry derived from previous PA-RISC microprocessors. The PA-8000 was introduced on |
| 36 | 2157 | `' a'` | latin_word | 35/35 | 47.3 | National Digital Television Center in Centennial, Colorado as a wholly owned subsidiary, which is today known as the |
| 37 | 2437 | `' PA'` | latin_word | 35/35 | 39.1 | 0 processors (PA-8200 to PA-8900, described further below) |
| 38 | 2430 | `'PA'` | latin_word | 35/35 | 39.1 | follow-on PA-8x00 processors (PA-8200 to PA-89 |
| 39 | 2314 | `' code'` | latin_word | 35/35 | 45.8 | -8000 (PCX-U), code-named "Onyx", is a microprocessor |
| 40 | 2400 | `' ('` | punct_or_symbol | 35/35 | 44.1 | shipments began to members of the Precision RISC Organization (PRO). It was used exclusively by PRO members and |
| 41 | 2055 | `' the'` | latin_word | 35/35 | 55.3 | 44.5 billion. The proposed name for the merged company was "AT&T Comcast", but the |
| 42 | 2439 | `'8'` | number | 35/35 | 42.5 | (PA-8200 to PA-8900, described further below) are based |
| 43 | 2438 | `'-'` | punct_or_symbol | 35/35 | 42.7 | processors (PA-8200 to PA-8900, described further below) are |
| 44 | 2351 | `').'` | punct_or_symbol | 35/35 | 46.1 | ISC 2.0 instruction set architecture (ISA). It was a completely new design with no circuitry |
| 45 | 1532 | `' the'` | latin_word | 35/35 | 62.5 | ooks were able to claim individual distinctions such as being the lightest or thinnest, the Air was |
| 46 | 2425 | `'x'` | latin_word | 35/35 | 47.4 | the merchant market. All follow-on PA-8x00 processors (PA-8200 |
| 47 | 2440 | `'9'` | number | 35/35 | 48.0 | PA-8200 to PA-8900, described further below) are based on |
| 48 | 2432 | `'8'` | number | 35/35 | 51.3 | PA-8x00 processors (PA-8200 to PA-8900 |
| 49 | 2442 | `'0'` | number | 35/35 | 49.7 | 8200 to PA-8900, described further below) are based on the basic |
| 50 | 2431 | `'-'` | punct_or_symbol | 35/35 | 51.9 | -on PA-8x00 processors (PA-8200 to PA-890 |
| 51 | 2441 | `'0'` | number | 35/35 | 52.4 | -8200 to PA-8900, described further below) are based on the |
| 52 | 2420 | `' follow'` | latin_word | 35/35 | 54.2 | and was not sold on the merchant market. All follow-on PA-8x00 processors (PA |
| 53 | 2419 | `' All'` | latin_word | 35/35 | 57.6 | members and was not sold on the merchant market. All follow-on PA-8x00 processors ( |
| 54 | 2407 | `' by'` | latin_word | 35/35 | 57.3 | RISC Organization (PRO). It was used exclusively by PRO members and was not sold on the merchant market |
| 55 | 2433 | `'2'` | number | 35/35 | 58.1 | -8x00 processors (PA-8200 to PA-8900, |
| 56 | 2405 | `' used'` | latin_word | 35/35 | 60.5 | the Precision RISC Organization (PRO). It was used exclusively by PRO members and was not sold on the |
| 57 | 2426 | `'0'` | number | 35/35 | 60.7 | merchant market. All follow-on PA-8x00 processors (PA-8200 to |
| 58 | 2421 | `'-on'` | other_text | 35/35 | 60.8 | was not sold on the merchant market. All follow-on PA-8x00 processors (PA- |
| 59 | 2435 | `'0'` | number | 35/35 | 60.9 | x00 processors (PA-8200 to PA-8900, described further |
| 60 | 2434 | `'0'` | number | 35/35 | 62.9 | 8x00 processors (PA-8200 to PA-8900, described |
| 61 | 2406 | `' exclusively'` | latin_word | 35/35 | 65.7 | Precision RISC Organization (PRO). It was used exclusively by PRO members and was not sold on the merchant |
| 62 | 2422 | `' PA'` | latin_word | 35/35 | 62.9 | not sold on the merchant market. All follow-on PA-8x00 processors (PA-8 |
| 63 | 2413 | `' sold'` | latin_word | 35/35 | 68.4 | It was used exclusively by PRO members and was not sold on the merchant market. All follow-on PA- |
| 64 | 2404 | `' was'` | latin_word | 35/35 | 67.0 | of the Precision RISC Organization (PRO). It was used exclusively by PRO members and was not sold on |
| 65 | 2423 | `'-'` | punct_or_symbol | 35/35 | 68.0 | sold on the merchant market. All follow-on PA-8x00 processors (PA-82 |
| 66 | 2320 | `'",'` | punct_or_symbol | 35/35 | 69.8 | PCX-U), code-named "Onyx", is a microprocessor developed and fabricated by Hewlett |
| 67 | 2382 | `'2'` | number | 35/35 | 69.5 | PA-8000 was introduced on 2 November 1995 when shipments began to |
| 68 | 2427 | `'0'` | number | 35/35 | 70.6 | market. All follow-on PA-8x00 processors (PA-8200 to PA |
| 69 | 2247 | `' a'` | latin_word | 35/35 | 75.0 | base project founders and Membase, Inc. announced a merger with CouchOne (a company with many of |
| 70 | 2424 | `'8'` | number | 35/35 | 71.2 | on the merchant market. All follow-on PA-8x00 processors (PA-820 |
| 71 | 4 | `':'` | punct_or_symbol | 35/35 | 74.0 | Document: Macintosh: Compaq, who had previously held the third |
| 72 | 2371 | `' The'` | latin_word | 35/35 | 73.6 | ry derived from previous PA-RISC microprocessors. The PA-8000 was introduced on |
| 73 | 2411 | `' was'` | latin_word | 35/35 | 74.1 | PRO). It was used exclusively by PRO members and was not sold on the merchant market. All follow-on |
| 74 | 2409 | `' members'` | latin_word | 35/35 | 76.4 | Organization (PRO). It was used exclusively by PRO members and was not sold on the merchant market. All |
| 75 | 2403 | `' It'` | latin_word | 35/35 | 78.1 | members of the Precision RISC Organization (PRO). It was used exclusively by PRO members and was not sold |
| 76 | 2416 | `' merchant'` | latin_word | 35/35 | 78.2 | exclusively by PRO members and was not sold on the merchant market. All follow-on PA-8x0 |
| 77 | 2338 | `' the'` | latin_word | 35/35 | 78.7 | by Hewlett-Packard (HP) that implemented the PA-RISC 2.0 instruction set architecture |
| 78 | 2414 | `' on'` | latin_word | 35/35 | 80.9 | was used exclusively by PRO members and was not sold on the merchant market. All follow-on PA-8 |
| 79 | 2362 | `' derived'` | latin_word | 35/35 | 87.5 | It was a completely new design with no circuitry derived from previous PA-RISC microprocessors. The PA |
| 80 | 2235 | `' the'` | latin_word | 35/35 | 95.0 | February 8, 2011, the Membase project founders and Membase, Inc. |

## Example 13

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2438 | `'.\n'` | punct_or_symbol | 35/35 | 1.1 | white cross for the red - on - white one.\n |
| 2 | 2360 | `' a'` | latin_word | 35/35 | 1.9 | England and Philip II of France agreed to go on a crusade, and that Henry would use a white |
| 3 | 2409 | `' vice'` | latin_word | 35/35 | 3.1 | and the French king the red one (and not vice versa as suggested by later use). It is not |
| 4 | 2431 | `' the'` | latin_word | 35/35 | 4.3 | at what point the English exchanged the white cross for the red - on - white one.\n |
| 5 | 2416 | `').'` | punct_or_symbol | 35/35 | 5.6 | (and not vice versa as suggested by later use). It is not clear at what point the English exchanged |
| 6 | 2437 | `' one'` | latin_word | 35/35 | 5.6 | the white cross for the red - on - white one.\n |
| 7 | 2433 | `' -'` | punct_or_symbol | 35/35 | 7.4 | point the English exchanged the white cross for the red - on - white one.\n |
| 8 | 2427 | `' the'` | latin_word | 35/35 | 9.9 | It is not clear at what point the English exchanged the white cross for the red - on - white one |
| 9 | 2434 | `' on'` | latin_word | 35/35 | 9.3 | the English exchanged the white cross for the red - on - white one.\n |
| 10 | 2435 | `' -'` | punct_or_symbol | 35/35 | 10.4 | English exchanged the white cross for the red - on - white one.\n |
| 11 | 2424 | `' the'` | latin_word | 35/35 | 10.1 | later use). It is not clear at what point the English exchanged the white cross for the red - on |
| 12 | 2436 | `' white'` | latin_word | 35/35 | 11.7 | exchanged the white cross for the red - on - white one.\n |
| 13 | 2426 | `' exchanged'` | latin_word | 35/35 | 14.4 | ). It is not clear at what point the English exchanged the white cross for the red - on - white |
| 14 | 2430 | `' for'` | latin_word | 35/35 | 15.6 | clear at what point the English exchanged the white cross for the red - on - white one.\n |
| 15 | 2388 | `' the'` | latin_word | 35/35 | 18.4 | 13th - century authorities are unanimous on the point that the English king adopted the white cross, |
| 16 | 2420 | `' clear'` | latin_word | 35/35 | 19.1 | versa as suggested by later use). It is not clear at what point the English exchanged the white cross for |
| 17 | 2398 | `','` | punct_or_symbol | 35/35 | 19.3 | the point that the English king adopted the white cross, and the French king the red one (and not |
| 18 | 2422 | `' what'` | latin_word | 35/35 | 19.9 | suggested by later use). It is not clear at what point the English exchanged the white cross for the red |
| 19 | 376 | `' '` | space | 35/35 | 29.0 | 1540 – 26 January 1568), born Lady Katherine Grey, |
| 20 | 2432 | `' red'` | latin_word | 35/35 | 20.7 | what point the English exchanged the white cross for the red - on - white one.\n |
| 21 | 2425 | `' English'` | latin_word | 35/35 | 22.0 | use). It is not clear at what point the English exchanged the white cross for the red - on - |
| 22 | 2423 | `' point'` | latin_word | 35/35 | 23.0 | by later use). It is not clear at what point the English exchanged the white cross for the red - |
| 23 | 2418 | `' is'` | latin_word | 35/35 | 23.3 | not vice versa as suggested by later use). It is not clear at what point the English exchanged the white |
| 24 | 2417 | `' It'` | latin_word | 35/35 | 26.1 | and not vice versa as suggested by later use). It is not clear at what point the English exchanged the |
| 25 | 2377 | `'.'` | punct_or_symbol | 35/35 | 26.7 | would use a white cross and Philip a red cross. 13th - century authorities are unanimous on |
| 26 | 2419 | `' not'` | latin_word | 35/35 | 28.1 | vice versa as suggested by later use). It is not clear at what point the English exchanged the white cross |
| 27 | 2428 | `' white'` | latin_word | 35/35 | 27.5 | is not clear at what point the English exchanged the white cross for the red - on - white one.\n |
| 28 | 2421 | `' at'` | latin_word | 35/35 | 30.4 | as suggested by later use). It is not clear at what point the English exchanged the white cross for the |
| 29 | 2429 | `' cross'` | latin_word | 35/35 | 30.7 | not clear at what point the English exchanged the white cross for the red - on - white one.\n |
| 30 | 2308 | `' here'` | latin_word | 35/35 | 32.1 | translated. The New English Bible was also put together here in the 20th century. Westminster suffered |
| 31 | 2406 | `' ('` | punct_or_symbol | 35/35 | 31.2 | white cross, and the French king the red one (and not vice versa as suggested by later use). |
| 32 | 2369 | `' a'` | latin_word | 35/35 | 35.1 | on a crusade, and that Henry would use a white cross and Philip a red cross. 1 |
| 33 | 2322 | `' the'` | latin_word | 35/35 | 32.5 | 20th century. Westminster suffered minor damage during the Blitz on 15 November 194 |
| 34 | 2391 | `' the'` | latin_word | 35/35 | 35.0 | th - century authorities are unanimous on the point that the English king adopted the white cross, and the French |
| 35 | 2395 | `' the'` | latin_word | 35/35 | 35.7 | are unanimous on the point that the English king adopted the white cross, and the French king the red one |
| 36 | 2315 | `' century'` | latin_word | 35/35 | 41.8 | also put together here in the 20th century. Westminster suffered minor damage during the Blitz on |
| 37 | 2415 | `' use'` | latin_word | 35/35 | 37.9 | one (and not vice versa as suggested by later use). It is not clear at what point the English |
| 38 | 377 | `'1'` | number | 35/35 | 45.8 | 1540 – 26 January 1568), born Lady Katherine Grey, was |
| 39 | 2403 | `' the'` | latin_word | 35/35 | 38.8 | king adopted the white cross, and the French king the red one (and not vice versa as suggested by |
| 40 | 2384 | `' authorities'` | latin_word | 35/35 | 39.3 | a red cross. 13th - century authorities are unanimous on the point that the English king adopted |
| 41 | 2254 | `' '` | space | 35/35 | 52.7 | for himself.\nDocument: Westminster Abbey: Until the 19th century, Westminster was the third seat |
| 42 | 2334 | `'.\n'` | punct_or_symbol | 35/35 | 42.4 | on 15 November 1940.\nDocument: Flag of England: In 11 |
| 43 | 2413 | `' by'` | latin_word | 35/35 | 43.0 | the red one (and not vice versa as suggested by later use). It is not clear at what point |
| 44 | 2412 | `' suggested'` | latin_word | 35/35 | 44.6 | king the red one (and not vice versa as suggested by later use). It is not clear at what |
| 45 | 2414 | `' later'` | latin_word | 35/35 | 47.3 | red one (and not vice versa as suggested by later use). It is not clear at what point the |
| 46 | 2345 | `'8'` | number | 35/35 | 51.3 | Document: Flag of England: In 1188 Henry II of England and Philip II of France |
| 47 | 2400 | `' the'` | latin_word | 35/35 | 47.9 | that the English king adopted the white cross, and the French king the red one (and not vice versa |
| 48 | 2407 | `'and'` | latin_word | 35/35 | 48.7 | cross, and the French king the red one (and not vice versa as suggested by later use). It |
| 49 | 2401 | `' French'` | latin_word | 35/35 | 50.2 | the English king adopted the white cross, and the French king the red one (and not vice versa as |
| 50 | 2385 | `' are'` | latin_word | 35/35 | 50.1 | red cross. 13th - century authorities are unanimous on the point that the English king adopted the |
| 51 | 2410 | `' versa'` | latin_word | 35/35 | 52.1 | the French king the red one (and not vice versa as suggested by later use). It is not clear |
| 52 | 2382 | `' -'` | punct_or_symbol | 35/35 | 52.2 | and Philip a red cross. 13th - century authorities are unanimous on the point that the English |
| 53 | 2411 | `' as'` | latin_word | 35/35 | 52.9 | French king the red one (and not vice versa as suggested by later use). It is not clear at |
| 54 | 2399 | `' and'` | latin_word | 35/35 | 52.8 | point that the English king adopted the white cross, and the French king the red one (and not vice |
| 55 | 2363 | `','` | punct_or_symbol | 35/35 | 54.0 | II of France agreed to go on a crusade, and that Henry would use a white cross and Philip |
| 56 | 2340 | `':'` | punct_or_symbol | 35/35 | 54.8 | 1940.\nDocument: Flag of England: In 1188 Henry II of England |
| 57 | 2408 | `' not'` | latin_word | 35/35 | 57.7 | , and the French king the red one (and not vice versa as suggested by later use). It is |
| 58 | 2310 | `' the'` | latin_word | 35/35 | 61.4 | The New English Bible was also put together here in the 20th century. Westminster suffered minor damage |
| 59 | 2402 | `' king'` | latin_word | 35/35 | 58.7 | English king adopted the white cross, and the French king the red one (and not vice versa as suggested |
| 60 | 2390 | `' that'` | latin_word | 35/35 | 59.9 | 3th - century authorities are unanimous on the point that the English king adopted the white cross, and the |
| 61 | 2378 | `' '` | space | 35/35 | 60.5 | use a white cross and Philip a red cross. 13th - century authorities are unanimous on the |
| 62 | 2405 | `' one'` | latin_word | 35/35 | 64.9 | the white cross, and the French king the red one (and not vice versa as suggested by later use |
| 63 | 2383 | `' century'` | latin_word | 35/35 | 66.0 | Philip a red cross. 13th - century authorities are unanimous on the point that the English king |
| 64 | 2404 | `' red'` | latin_word | 35/35 | 68.2 | adopted the white cross, and the French king the red one (and not vice versa as suggested by later |
| 65 | 2389 | `' point'` | latin_word | 35/35 | 70.0 | 13th - century authorities are unanimous on the point that the English king adopted the white cross, and |
| 66 | 2393 | `' king'` | latin_word | 35/35 | 70.9 | century authorities are unanimous on the point that the English king adopted the white cross, and the French king the |
| 67 | 2336 | `':'` | punct_or_symbol | 35/35 | 73.0 | 15 November 1940.\nDocument: Flag of England: In 1188 |
| 68 | 2386 | `' unanimous'` | latin_word | 35/35 | 73.3 | cross. 13th - century authorities are unanimous on the point that the English king adopted the white |
| 69 | 2394 | `' adopted'` | latin_word | 35/35 | 74.3 | authorities are unanimous on the point that the English king adopted the white cross, and the French king the red |
| 70 | 2372 | `' and'` | latin_word | 35/35 | 74.8 | ade, and that Henry would use a white cross and Philip a red cross. 13th - |
| 71 | 2355 | `' France'` | latin_word | 35/35 | 77.8 | 88 Henry II of England and Philip II of France agreed to go on a crusade, and that |
| 72 | 2392 | `' English'` | latin_word | 35/35 | 77.5 | - century authorities are unanimous on the point that the English king adopted the white cross, and the French king |
| 73 | 1499 | `' to'` | latin_word | 35/35 | 102.4 | fought successive, increasingly expensive, campaigns in a bid to regain these possessions. John's efforts to raise revenues |
| 74 | 2387 | `' on'` | latin_word | 35/35 | 78.0 | . 13th - century authorities are unanimous on the point that the English king adopted the white cross |
| 75 | 2316 | `'.'` | punct_or_symbol | 35/35 | 80.8 | put together here in the 20th century. Westminster suffered minor damage during the Blitz on 1 |
| 76 | 2379 | `'1'` | number | 35/35 | 80.1 | a white cross and Philip a red cross. 13th - century authorities are unanimous on the point |
| 77 | 2246 | `'.\n'` | punct_or_symbol | 35/35 | 85.1 | was able to seize the crown of England for himself.\nDocument: Westminster Abbey: Until the 19 |
| 78 | 2374 | `' a'` | latin_word | 35/35 | 87.4 | and that Henry would use a white cross and Philip a red cross. 13th - century authorities |
| 79 | 2174 | `'.\n'` | punct_or_symbol | 35/35 | 86.4 | convinced Richard to allow John into England in his absence.\nDocument: Robert Curthose: When William II died |
| 80 | 2299 | `'.'` | punct_or_symbol | 35/35 | 88.1 | and the last half of the New Testament were translated. The New English Bible was also put together here in |

## Example 14

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2119 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | are the parents of actress Condola Rashād.\n |
| 2 | 1244 | `'1'` | number | 35/35 | 2.7 | 974) and Lyda Bunker (1889–1955). She |
| 3 | 2110 | `' the'` | latin_word | 35/35 | 2.9 | . Phylicia and Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 4 | 2100 | `'.'` | punct_or_symbol | 35/35 | 4.2 | player turned sportscaster, Ahmad Rashād. Phylicia and Ahmad Rashād are the |
| 5 | 2113 | `' actress'` | latin_word | 35/35 | 7.4 | ia and Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 6 | 2109 | `' are'` | latin_word | 35/35 | 6.9 | d. Phylicia and Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 7 | 2104 | `' and'` | latin_word | 35/35 | 7.6 | aster, Ahmad Rashād. Phylicia and Ahmad Rashād are the parents of actress Cond |
| 8 | 2081 | `' the'` | latin_word | 35/35 | 9.8 | spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned |
| 9 | 2118 | `'d'` | latin_word | 35/35 | 10.2 | d are the parents of actress Condola Rashād.\n |
| 10 | 2111 | `' parents'` | latin_word | 35/35 | 10.7 | Phylicia and Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 11 | 2116 | `' Rash'` | latin_word | 35/35 | 11.8 | Rashād are the parents of actress Condola Rashād.\n |
| 12 | 2112 | `' of'` | latin_word | 35/35 | 12.5 | licia and Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 13 | 2115 | `'ola'` | latin_word | 35/35 | 12.3 | Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 14 | 2114 | `' Cond'` | latin_word | 35/35 | 13.9 | and Ahmad Rashād are the parents of actress Condola Rashād.\n |
| 15 | 2117 | `'ā'` | other_text | 35/35 | 16.9 | ād are the parents of actress Condola Rashād.\n |
| 16 | 2016 | `' a'` | latin_word | 35/35 | 16.8 | Blair High School, have formed Oakdale Pictures, a production company in Reno.\nDocument: List of show |
| 17 | 2095 | `','` | punct_or_symbol | 35/35 | 17.9 | , and former NFL football player turned sportscaster, Ahmad Rashād. Phylicia and Ahmad |
| 18 | 2069 | `' the'` | latin_word | 35/35 | 18.3 | Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer |
| 19 | 2061 | `'.'` | punct_or_symbol | 35/35 | 21.7 | is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse |
| 20 | 2085 | `','` | punct_or_symbol | 35/35 | 23.9 | Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, |
| 21 | 2076 | `','` | punct_or_symbol | 35/35 | 24.2 | ād is the former spouse of both Victor Willis, former lead singer of the group Village People, and |
| 22 | 323 | `' a'` | latin_word | 35/35 | 32.9 | to be the real Santa. The film has become a perennial Christmas favorite.\nDocument: Miracle on 3 |
| 23 | 2091 | `' turned'` | latin_word | 35/35 | 27.9 | the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phy |
| 24 | 2096 | `' Ahmad'` | latin_word | 35/35 | 27.1 | and former NFL football player turned sportscaster, Ahmad Rashād. Phylicia and Ahmad Rash |
| 25 | 1213 | `'9'` | number | 35/35 | 45.5 | Hunt was born on January 8, 1923, the daughter of oilman H. |
| 26 | 1994 | `'9'` | number | 35/35 | 39.7 | Brandon R. MacDuff (born 1954), a 1972 graduate |
| 27 | 2105 | `' Ahmad'` | latin_word | 35/35 | 28.4 | , Ahmad Rashād. Phylicia and Ahmad Rashād are the parents of actress Condola |
| 28 | 2042 | `' the'` | latin_word | 35/35 | 30.8 | director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married |
| 29 | 1109 | `'-'` | punct_or_symbol | 35/35 | 39.8 | sister of Natalie Wood.\nDocument: Bombers B-52 (released in the UK as No Sleep |
| 30 | 2101 | `' Phy'` | latin_word | 35/35 | 31.6 | turned sportscaster, Ahmad Rashād. Phylicia and Ahmad Rashād are the parents |
| 31 | 2058 | `','` | punct_or_symbol | 35/35 | 32.3 | Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is |
| 32 | 1587 | `' alone'` | latin_word | 35/35 | 39.7 | ard). During a thunderstorm Novalee, alone at Walmart, goes into labor. Forney, |
| 33 | 2103 | `'ia'` | latin_word | 35/35 | 38.9 | scaster, Ahmad Rashād. Phylicia and Ahmad Rashād are the parents of actress |
| 34 | 2106 | `' Rash'` | latin_word | 35/35 | 38.4 | Ahmad Rashād. Phylicia and Ahmad Rashād are the parents of actress Condola Rash |
| 35 | 2021 | `'.\n'` | punct_or_symbol | 35/35 | 37.3 | formed Oakdale Pictures, a production company in Reno.\nDocument: List of show business families: Actress / |
| 36 | 2072 | `' of'` | latin_word | 35/35 | 39.7 | Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group |
| 37 | 2086 | `' and'` | latin_word | 35/35 | 38.9 | , former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad |
| 38 | 2108 | `'d'` | latin_word | 35/35 | 41.7 | ād. Phylicia and Ahmad Rashād are the parents of actress Condola Rashād |
| 39 | 2088 | `' NFL'` | latin_word | 35/35 | 41.5 | lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashā |
| 40 | 2049 | `','` | punct_or_symbol | 35/35 | 40.7 | ād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm |
| 41 | 1987 | `' Mac'` | latin_word | 35/35 | 56.6 | : He and his older brother, Brandon R. MacDuff (born 1954), |
| 42 | 2107 | `'ā'` | other_text | 35/35 | 41.8 | Rashād. Phylicia and Ahmad Rashād are the parents of actress Condola Rashā |
| 43 | 2092 | `' sport'` | latin_word | 35/35 | 43.6 | group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phylic |
| 44 | 2098 | `'ā'` | other_text | 35/35 | 44.9 | NFL football player turned sportscaster, Ahmad Rashād. Phylicia and Ahmad Rashād |
| 45 | 2102 | `'lic'` | latin_word | 35/35 | 45.5 | sportscaster, Ahmad Rashād. Phylicia and Ahmad Rashād are the parents of |
| 46 | 2097 | `' Rash'` | latin_word | 35/35 | 47.1 | former NFL football player turned sportscaster, Ahmad Rashād. Phylicia and Ahmad Rashā |
| 47 | 1602 | `' awe'` | latin_word | 35/35 | 77.5 | into labor. Forney, who is now in awe of Novalee, smashes through the Walmart |
| 48 | 2094 | `'aster'` | latin_word | 35/35 | 52.1 | People, and former NFL football player turned sportscaster, Ahmad Rashād. Phylicia and |
| 49 | 2068 | `' is'` | latin_word | 35/35 | 55.2 | , Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead |
| 50 | 1754 | `' the'` | latin_word | 35/35 | 72.8 | still - standing buckeye tree amidst the damage from the storm. After the funeral, Novalee finds |
| 51 | 2082 | `' group'` | latin_word | 35/35 | 56.6 | of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sport |
| 52 | 2089 | `' football'` | latin_word | 35/35 | 56.2 | singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād |
| 53 | 2083 | `' Village'` | latin_word | 35/35 | 57.8 | both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportsc |
| 54 | 679 | `' the'` | latin_word | 35/35 | 63.5 | to find that the man assigned to play Santa in the annual Macy's Thanksgiving Day Parade (Percy |
| 55 | 2090 | `' player'` | latin_word | 35/35 | 60.0 | of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. |
| 56 | 2099 | `'d'` | latin_word | 35/35 | 59.1 | football player turned sportscaster, Ahmad Rashād. Phylicia and Ahmad Rashād are |
| 57 | 2079 | `' singer'` | latin_word | 35/35 | 59.5 | the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football |
| 58 | 2084 | `' People'` | latin_word | 35/35 | 59.9 | Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster |
| 59 | 1108 | `' B'` | latin_word | 35/35 | 66.1 | the sister of Natalie Wood.\nDocument: Bombers B-52 (released in the UK as No |
| 60 | 2093 | `'sc'` | latin_word | 35/35 | 61.8 | Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phylicia |
| 61 | 2071 | `' spouse'` | latin_word | 35/35 | 67.7 | . Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the |
| 62 | 2087 | `' former'` | latin_word | 35/35 | 69.3 | former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rash |
| 63 | 970 | `' '` | space | 35/35 | 80.3 | 7, 1915 -- November 14, 2002) was |
| 64 | 2075 | `' Willis'` | latin_word | 35/35 | 69.9 | Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, |
| 65 | 2080 | `' of'` | latin_word | 35/35 | 70.1 | former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player |
| 66 | 2029 | `':'` | punct_or_symbol | 35/35 | 70.3 | in Reno.\nDocument: List of show business families: Actress / director / singer Phylicia Rashā |
| 67 | 1970 | `'.\n'` | punct_or_symbol | 35/35 | 70.9 | Katie role was created circa 1966.\nDocument: Dana MacDuff: He and his |
| 68 | 2001 | `'9'` | number | 35/35 | 87.8 | born 1954), a 1972 graduate of Blair High School, have formed |
| 69 | 2027 | `' business'` | latin_word | 35/35 | 75.9 | production company in Reno.\nDocument: List of show business families: Actress / director / singer Phylicia |
| 70 | 2015 | `','` | punct_or_symbol | 35/35 | 76.8 | of Blair High School, have formed Oakdale Pictures, a production company in Reno.\nDocument: List of |
| 71 | 2060 | `' Nixon'` | latin_word | 35/35 | 77.5 | who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former |
| 72 | 1870 | `' the'` | latin_word | 35/35 | 76.5 | she previously had a recurring role as Sunny Day in the detective series Hawaiian Eye (1963). |
| 73 | 1992 | `' '` | space | 35/35 | 87.1 | brother, Brandon R. MacDuff (born 1954), a 197 |
| 74 | 2009 | `','` | punct_or_symbol | 35/35 | 83.0 | 1972 graduate of Blair High School, have formed Oakdale Pictures, a production company in |
| 75 | 2073 | `' both'` | latin_word | 35/35 | 83.2 | licia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village |
| 76 | 2041 | `' is'` | latin_word | 35/35 | 83.6 | / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is |
| 77 | 2023 | `':'` | punct_or_symbol | 35/35 | 83.3 | dale Pictures, a production company in Reno.\nDocument: List of show business families: Actress / director / |
| 78 | 2059 | `' Norm'` | latin_word | 35/35 | 89.3 | , who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the |
| 79 | 2078 | `' lead'` | latin_word | 35/35 | 90.3 | is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL |
| 80 | 1592 | `' into'` | latin_word | 35/35 | 99.1 | storm Novalee, alone at Walmart, goes into labor. Forney, who is now in awe |

## Example 15

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1550 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | -- with a population of 2.3 million.\n |
| 2 | 1487 | `' the'` | latin_word | 35/35 | 2.9 | Colloquially referred to as Greater Houston, the 10,000 - square - |
| 3 | 1545 | `' '` | space | 35/35 | 3.2 | cultural center of the South -- with a population of 2.3 million.\n |
| 4 | 1030 | `' '` | space | 35/35 | 5.7 | of four National Forests in Texas, is located 50 miles north of Houston. The forest is |
| 5 | 1547 | `'.'` | punct_or_symbol | 35/35 | 4.7 | of the South -- with a population of 2.3 million.\n |
| 6 | 1546 | `'2'` | number | 35/35 | 5.2 | center of the South -- with a population of 2.3 million.\n |
| 7 | 1549 | `' million'` | latin_word | 35/35 | 6.5 | South -- with a population of 2.3 million.\n |
| 8 | 1538 | `' the'` | latin_word | 35/35 | 8.5 | of Houston -- the largest economic and cultural center of the South -- with a population of 2.3 |
| 9 | 1542 | `' a'` | latin_word | 35/35 | 9.1 | largest economic and cultural center of the South -- with a population of 2.3 million.\n |
| 10 | 1548 | `'3'` | number | 35/35 | 10.5 | the South -- with a population of 2.3 million.\n |
| 11 | 1521 | `' the'` | latin_word | 35/35 | 11.5 | around Harris County, the third-most populous county in the nation, which contains the city of Houston -- the |
| 12 | 1526 | `' the'` | latin_word | 35/35 | 14.9 | third-most populous county in the nation, which contains the city of Houston -- the largest economic and cultural center |
| 13 | 904 | `' '` | space | 35/35 | 18.9 | of West Texas A&M University. The stadium holds 20,000 people and was built |
| 14 | 1544 | `' of'` | latin_word | 35/35 | 14.1 | and cultural center of the South -- with a population of 2.3 million.\n |
| 15 | 1543 | `' population'` | latin_word | 35/35 | 14.7 | economic and cultural center of the South -- with a population of 2.3 million.\n |
| 16 | 1541 | `' with'` | latin_word | 35/35 | 15.9 | the largest economic and cultural center of the South -- with a population of 2.3 million.\n |
| 17 | 787 | `'2'` | number | 35/35 | 18.9 | 13,303 at the 2010 census. It is part of the |
| 18 | 1514 | `','` | punct_or_symbol | 35/35 | 19.8 | 00 km) region is centered around Harris County, the third-most populous county in the nation, which |
| 19 | 1540 | `' --'` | punct_or_symbol | 35/35 | 19.2 | -- the largest economic and cultural center of the South -- with a population of 2.3 million.\n |
| 20 | 1534 | `' and'` | latin_word | 35/35 | 20.2 | which contains the city of Houston -- the largest economic and cultural center of the South -- with a population of |
| 21 | 1523 | `','` | punct_or_symbol | 35/35 | 20.0 | County, the third-most populous county in the nation, which contains the city of Houston -- the largest economic |
| 22 | 468 | `'0'` | number | 35/35 | 24.5 | 5,464 at the 2000 census. It is named for Charles C |
| 23 | 1531 | `' the'` | latin_word | 35/35 | 22.7 | the nation, which contains the city of Houston -- the largest economic and cultural center of the South -- with |
| 24 | 1530 | `' --'` | punct_or_symbol | 35/35 | 24.3 | in the nation, which contains the city of Houston -- the largest economic and cultural center of the South -- |
| 25 | 1533 | `' economic'` | latin_word | 35/35 | 25.3 | , which contains the city of Houston -- the largest economic and cultural center of the South -- with a population |
| 26 | 1515 | `' the'` | latin_word | 35/35 | 29.5 | 0 km) region is centered around Harris County, the third-most populous county in the nation, which contains |
| 27 | 1527 | `' city'` | latin_word | 35/35 | 33.0 | -most populous county in the nation, which contains the city of Houston -- the largest economic and cultural center of |
| 28 | 1532 | `' largest'` | latin_word | 35/35 | 31.6 | nation, which contains the city of Houston -- the largest economic and cultural center of the South -- with a |
| 29 | 1507 | `')'` | punct_or_symbol | 35/35 | 34.2 | - mile (26,000 km) region is centered around Harris County, the third-most |
| 30 | 98 | `' and'` | latin_word | 35/35 | 40.9 | .\nDocument: Little Pine Creek (Luzerne and Columbia Counties, Pennsylvania): Little Pine Creek is a |
| 31 | 1536 | `' center'` | latin_word | 35/35 | 34.5 | the city of Houston -- the largest economic and cultural center of the South -- with a population of 2 |
| 32 | 1476 | `'.'` | punct_or_symbol | 35/35 | 35.1 | counties along the Gulf Coast of the state of Texas. Colloquially referred to as Greater Houston, |
| 33 | 1468 | `' the'` | latin_word | 35/35 | 41.1 | in the United States, encompassing nine counties along the Gulf Coast of the state of Texas. Collo |
| 34 | 1535 | `' cultural'` | latin_word | 35/35 | 39.1 | contains the city of Houston -- the largest economic and cultural center of the South -- with a population of |
| 35 | 1525 | `' contains'` | latin_word | 35/35 | 37.2 | the third-most populous county in the nation, which contains the city of Houston -- the largest economic and cultural |
| 36 | 1522 | `' nation'` | latin_word | 35/35 | 38.5 | Harris County, the third-most populous county in the nation, which contains the city of Houston -- the largest |
| 37 | 1539 | `' South'` | latin_word | 35/35 | 37.1 | Houston -- the largest economic and cultural center of the South -- with a population of 2.3 million |
| 38 | 797 | `' the'` | latin_word | 35/35 | 42.7 | 2010 census. It is part of the Amarillo, Texas, metropolitan statistical area. Canyon |
| 39 | 1502 | `','` | punct_or_symbol | 35/35 | 42.0 | 000 - square - mile (26,000 km) region is centered around Harris |
| 40 | 1518 | `' populous'` | latin_word | 35/35 | 40.4 | region is centered around Harris County, the third-most populous county in the nation, which contains the city of |
| 41 | 1529 | `' Houston'` | latin_word | 35/35 | 41.7 | county in the nation, which contains the city of Houston -- the largest economic and cultural center of the South |
| 42 | 1537 | `' of'` | latin_word | 35/35 | 42.8 | city of Houston -- the largest economic and cultural center of the South -- with a population of 2. |
| 43 | 1459 | `' the'` | latin_word | 35/35 | 43.3 | most populous metropolitan statistical area (MSA) in the United States, encompassing nine counties along the Gulf |
| 44 | 1524 | `' which'` | latin_word | 35/35 | 44.8 | , the third-most populous county in the nation, which contains the city of Houston -- the largest economic and |
| 45 | 1528 | `' of'` | latin_word | 35/35 | 46.7 | populous county in the nation, which contains the city of Houston -- the largest economic and cultural center of the |
| 46 | 1519 | `' county'` | latin_word | 35/35 | 47.5 | is centered around Harris County, the third-most populous county in the nation, which contains the city of Houston |
| 47 | 1500 | `'2'` | number | 35/35 | 49.5 | 0,000 - square - mile (26,000 km) region is centered |
| 48 | 1520 | `' in'` | latin_word | 35/35 | 50.3 | centered around Harris County, the third-most populous county in the nation, which contains the city of Houston -- |
| 49 | 1486 | `','` | punct_or_symbol | 35/35 | 50.5 | . Colloquially referred to as Greater Houston, the 10,000 - square |
| 50 | 1517 | `'-most'` | other_text | 35/35 | 53.3 | ) region is centered around Harris County, the third-most populous county in the nation, which contains the city |
| 51 | 1339 | `' the'` | latin_word | 35/35 | 55.4 | Grass Association. He died some three months prior to the passing of Smokey.\nDocument: Washburn |
| 52 | 1499 | `' ('` | punct_or_symbol | 35/35 | 53.4 | 10,000 - square - mile (26,000 km) region is |
| 53 | 1509 | `' is'` | latin_word | 35/35 | 54.3 | (26,000 km) region is centered around Harris County, the third-most populous county |
| 54 | 1435 | `'.\n'` | punct_or_symbol | 35/35 | 55.5 | burn is part of the Amarillo Metropolitan Statistical Area.\nDocument: Houston -- The Woodlands -- Sugar Land |
| 55 | 1501 | `'6'` | number | 35/35 | 57.7 | ,000 - square - mile (26,000 km) region is centered around |
| 56 | 1429 | `' the'` | latin_word | 35/35 | 57.5 | 2000. Washburn is part of the Amarillo Metropolitan Statistical Area.\nDocument: Houston -- |
| 57 | 1447 | `' the'` | latin_word | 35/35 | 60.2 | : Houston -- The Woodlands -- Sugar Land is the fifth most populous metropolitan statistical area (MSA) |
| 58 | 1512 | `' Harris'` | latin_word | 35/35 | 59.8 | ,000 km) region is centered around Harris County, the third-most populous county in the nation |
| 59 | 1457 | `')'` | punct_or_symbol | 35/35 | 64.6 | the fifth most populous metropolitan statistical area (MSA) in the United States, encompassing nine counties along |
| 60 | 1450 | `' populous'` | latin_word | 35/35 | 67.5 | The Woodlands -- Sugar Land is the fifth most populous metropolitan statistical area (MSA) in the United |
| 61 | 1511 | `' around'` | latin_word | 35/35 | 66.7 | 6,000 km) region is centered around Harris County, the third-most populous county in the |
| 62 | 1491 | `','` | punct_or_symbol | 35/35 | 68.2 | referred to as Greater Houston, the 10,000 - square - mile (26 |
| 63 | 1513 | `' County'` | latin_word | 35/35 | 68.4 | 000 km) region is centered around Harris County, the third-most populous county in the nation, |
| 64 | 1480 | `'ially'` | latin_word | 35/35 | 69.1 | Coast of the state of Texas. Colloquially referred to as Greater Houston, the 10 |
| 65 | 1508 | `' region'` | latin_word | 35/35 | 68.9 | mile (26,000 km) region is centered around Harris County, the third-most populous |
| 66 | 1488 | `' '` | space | 35/35 | 72.1 | loquially referred to as Greater Houston, the 10,000 - square - mile |
| 67 | 1495 | `' -'` | punct_or_symbol | 35/35 | 71.3 | Houston, the 10,000 - square - mile (26,000 |
| 68 | 1516 | `' third'` | latin_word | 35/35 | 71.9 | km) region is centered around Harris County, the third-most populous county in the nation, which contains the |
| 69 | 1462 | `','` | punct_or_symbol | 35/35 | 72.4 | statistical area (MSA) in the United States, encompassing nine counties along the Gulf Coast of the |
| 70 | 1510 | `' centered'` | latin_word | 35/35 | 73.1 | 26,000 km) region is centered around Harris County, the third-most populous county in |
| 71 | 1355 | `' is'` | latin_word | 35/35 | 106.5 | .\nDocument: Washburn, Texas: Washburn is an unincorporated community in Armstrong County, |
| 72 | 1454 | `' ('` | punct_or_symbol | 35/35 | 79.0 | Sugar Land is the fifth most populous metropolitan statistical area (MSA) in the United States, encompassing |
| 73 | 1211 | `' the'` | latin_word | 35/35 | 100.0 | ided in Hutchinson County near Spearman, which is the seat of Hansford County in the northern Panhandle |
| 74 | 1506 | `' km'` | latin_word | 35/35 | 79.7 | square - mile (26,000 km) region is centered around Harris County, the third |
| 75 | 1440 | `' The'` | latin_word | 35/35 | 81.6 | Amarillo Metropolitan Statistical Area.\nDocument: Houston -- The Woodlands -- Sugar Land is the fifth most populous |
| 76 | 1472 | `' the'` | latin_word | 35/35 | 81.5 | , encompassing nine counties along the Gulf Coast of the state of Texas. Colloquially referred to |
| 77 | 1446 | `' is'` | latin_word | 35/35 | 87.5 | Document: Houston -- The Woodlands -- Sugar Land is the fifth most populous metropolitan statistical area (MSA |
| 78 | 1498 | `' mile'` | latin_word | 35/35 | 87.3 | 10,000 - square - mile (26,000 km) region |
| 79 | 1345 | `'.\n'` | punct_or_symbol | 35/35 | 84.9 | three months prior to the passing of Smokey.\nDocument: Washburn, Texas: Washburn is |
| 80 | 1388 | `','` | punct_or_symbol | 35/35 | 88.5 | Highway 287 in northwestern Armstrong County, approximately 20 miles east of Amarillo. |

## Example 16

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2161 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | of the Client–server model helped to launch Ethernet.\n |
| 2 | 2160 | `' Ethernet'` | latin_word | 35/35 | 2.7 | vision of the Client–server model helped to launch Ethernet.\n |
| 3 | 2129 | `'2'` | number | 35/35 | 4.6 | and missed the opportunity to fund Lotus 1-2-3 or Visicalc. He also missed |
| 4 | 1775 | `'D'` | latin_word | 35/35 | 5.6 | 22%, as well as reduce the "Dell Hell" prominent on Internet search engines.\nDocument |
| 5 | 2140 | `' the'` | latin_word | 35/35 | 4.5 | -3 or Visicalc. He also missed the importance of the personal computer, but his futuristic vision |
| 6 | 2152 | `' the'` | latin_word | 35/35 | 5.7 | of the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 7 | 2158 | `' to'` | latin_word | 35/35 | 6.9 | his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 8 | 2159 | `' launch'` | latin_word | 35/35 | 7.5 | futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 9 | 2157 | `' helped'` | latin_word | 35/35 | 10.0 | but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 10 | 1641 | `' to'` | latin_word | 35/35 | 16.6 | cost-cutting "got in the way," aimed to reduce call transfer times and have call center representatives resolve |
| 11 | 2143 | `' the'` | latin_word | 35/35 | 12.7 | Visicalc. He also missed the importance of the personal computer, but his futuristic vision of the Client |
| 12 | 2146 | `','` | punct_or_symbol | 35/35 | 12.9 | . He also missed the importance of the personal computer, but his futuristic vision of the Client–server model |
| 13 | 2136 | `'.'` | punct_or_symbol | 35/35 | 13.7 | 1-2-3 or Visicalc. He also missed the importance of the personal computer, |
| 14 | 2154 | `'–'` | punct_or_symbol | 35/35 | 14.0 | personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 15 | 1247 | `'8'` | number | 35/35 | 22.0 | architecture (PC compatible).\nDocument: The PA-8000 (PCX-U), code-n |
| 16 | 2156 | `' model'` | latin_word | 35/35 | 16.5 | , but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 17 | 2126 | `' '` | space | 35/35 | 19.0 | from hardware, and missed the opportunity to fund Lotus 1-2-3 or Visicalc. |
| 18 | 2153 | `' Client'` | latin_word | 35/35 | 19.8 | the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 19 | 2149 | `' futuristic'` | latin_word | 35/35 | 20.3 | missed the importance of the personal computer, but his futuristic vision of the Client–server model helped to launch |
| 20 | 2155 | `'server'` | latin_word | 35/35 | 20.1 | computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 21 | 2130 | `'-'` | punct_or_symbol | 35/35 | 24.5 | missed the opportunity to fund Lotus 1-2-3 or Visicalc. He also missed the |
| 22 | 2148 | `' his'` | latin_word | 35/35 | 21.5 | also missed the importance of the personal computer, but his futuristic vision of the Client–server model helped to |
| 23 | 2150 | `' vision'` | latin_word | 35/35 | 23.2 | the importance of the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet |
| 24 | 2101 | `' the'` | latin_word | 35/35 | 23.5 | Development Corporation, and Apollo Computer. While believing in the value of software, he did not believe in the |
| 25 | 2147 | `' but'` | latin_word | 35/35 | 25.7 | He also missed the importance of the personal computer, but his futuristic vision of the Client–server model helped |
| 26 | 2138 | `' also'` | latin_word | 35/35 | 27.5 | -2-3 or Visicalc. He also missed the importance of the personal computer, but his |
| 27 | 2151 | `' of'` | latin_word | 35/35 | 27.5 | importance of the personal computer, but his futuristic vision of the Client–server model helped to launch Ethernet.\n |
| 28 | 2121 | `' the'` | latin_word | 35/35 | 28.6 | the value of software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 |
| 29 | 2111 | `' the'` | latin_word | 35/35 | 29.9 | the value of software, he did not believe in the value of software separate from hardware, and missed the |
| 30 | 2137 | `' He'` | latin_word | 35/35 | 30.3 | 1-2-3 or Visicalc. He also missed the importance of the personal computer, but |
| 31 | 2128 | `'-'` | punct_or_symbol | 35/35 | 33.2 | , and missed the opportunity to fund Lotus 1-2-3 or Visicalc. He also |
| 32 | 2048 | `' the'` | latin_word | 35/35 | 33.9 | of his own founding, Digital Equipment Corporation. At the time the book was published by two computer journal writers |
| 33 | 1717 | `'2'` | number | 35/35 | 35.3 | 2Dell blog, and then in February 2007, Michael Dell launched IdeaStorm.com |
| 34 | 2139 | `' missed'` | latin_word | 35/35 | 33.7 | 2-3 or Visicalc. He also missed the importance of the personal computer, but his futuristic |
| 35 | 2145 | `' computer'` | latin_word | 35/35 | 34.8 | c. He also missed the importance of the personal computer, but his futuristic vision of the Client–server |
| 36 | 2144 | `' personal'` | latin_word | 35/35 | 36.0 | icalc. He also missed the importance of the personal computer, but his futuristic vision of the Client– |
| 37 | 2141 | `' importance'` | latin_word | 35/35 | 37.6 | 3 or Visicalc. He also missed the importance of the personal computer, but his futuristic vision of |
| 38 | 2097 | `'.'` | punct_or_symbol | 35/35 | 38.2 | Symbolics, Lotus Development Corporation, and Apollo Computer. While believing in the value of software, he did |
| 39 | 2118 | `','` | punct_or_symbol | 35/35 | 38.6 | not believe in the value of software separate from hardware, and missed the opportunity to fund Lotus 1- |
| 40 | 2142 | `' of'` | latin_word | 35/35 | 40.8 | or Visicalc. He also missed the importance of the personal computer, but his futuristic vision of the |
| 41 | 2105 | `','` | punct_or_symbol | 35/35 | 43.1 | Apollo Computer. While believing in the value of software, he did not believe in the value of software separate |
| 42 | 2132 | `' or'` | latin_word | 35/35 | 43.4 | opportunity to fund Lotus 1-2-3 or Visicalc. He also missed the importance of |
| 43 | 2134 | `'ical'` | latin_word | 35/35 | 50.4 | fund Lotus 1-2-3 or Visicalc. He also missed the importance of the personal |
| 44 | 2127 | `'1'` | number | 35/35 | 54.4 | hardware, and missed the opportunity to fund Lotus 1-2-3 or Visicalc. He |
| 45 | 496 | `' '` | space | 35/35 | 59.1 | ard, 25 percent at Gateway, and 46 percent at Cisco. In 20 |
| 46 | 2124 | `' fund'` | latin_word | 35/35 | 51.6 | software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or Visical |
| 47 | 2133 | `' Vis'` | latin_word | 35/35 | 52.1 | to fund Lotus 1-2-3 or Visicalc. He also missed the importance of the |
| 48 | 1578 | `' the'` | latin_word | 35/35 | 88.2 | high customer satisfaction when PCs sold for thousands but by the 2000s, the company could |
| 49 | 2036 | `' the'` | latin_word | 35/35 | 57.3 | Ken Olsen racing to design minicomputers at the company of his own founding, Digital Equipment Corporation. |
| 50 | 2119 | `' and'` | latin_word | 35/35 | 54.0 | believe in the value of software separate from hardware, and missed the opportunity to fund Lotus 1-2 |
| 51 | 1422 | `','` | punct_or_symbol | 35/35 | 56.7 | 2 is a top down shooter game for IBM PC, developed by P-Squared Productions and released in |
| 52 | 1558 | `' the'` | latin_word | 35/35 | 61.1 | its technical support infrastructure, came under increasing scrutiny on the Web. The original Dell model was known for high |
| 53 | 1527 | `'2'` | number | 35/35 | 82.1 | Dell's reputation for poor customer service, since 2002, which was exacerbated as it moved |
| 54 | 2106 | `' he'` | latin_word | 35/35 | 57.5 | Computer. While believing in the value of software, he did not believe in the value of software separate from |
| 55 | 2135 | `'c'` | latin_word | 35/35 | 58.9 | Lotus 1-2-3 or Visicalc. He also missed the importance of the personal computer |
| 56 | 1829 | `' the'` | latin_word | 35/35 | 62.9 | 44.5 billion. The proposed name for the merged company was "AT&T Comcast", but the |
| 57 | 2131 | `'3'` | number | 35/35 | 60.2 | the opportunity to fund Lotus 1-2-3 or Visicalc. He also missed the importance |
| 58 | 985 | `'9'` | number | 35/35 | 85.4 | 0 specification, which was introduced in January 1996, defined data transfer rates of 1 |
| 59 | 2125 | `' Lotus'` | latin_word | 35/35 | 61.5 | separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or Visicalc |
| 60 | 2046 | `'.'` | punct_or_symbol | 35/35 | 62.7 | the company of his own founding, Digital Equipment Corporation. At the time the book was published by two computer |
| 61 | 2080 | `'),'` | punct_or_symbol | 35/35 | 64.5 | as Data General (founded by his former employee), Prime Computer, Wang Laboratories, Symbolics, Lotus |
| 62 | 1639 | `',"'` | punct_or_symbol | 35/35 | 79.3 | DNA of cost-cutting "got in the way," aimed to reduce call transfer times and have call center |
| 63 | 2109 | `' believe'` | latin_word | 35/35 | 67.1 | believing in the value of software, he did not believe in the value of software separate from hardware, and |
| 64 | 2099 | `' believing'` | latin_word | 35/35 | 67.8 | , Lotus Development Corporation, and Apollo Computer. While believing in the value of software, he did not believe |
| 65 | 2123 | `' to'` | latin_word | 35/35 | 68.2 | of software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or Vis |
| 66 | 2102 | `' value'` | latin_word | 35/35 | 77.6 | Corporation, and Apollo Computer. While believing in the value of software, he did not believe in the value |
| 67 | 2098 | `' While'` | latin_word | 35/35 | 74.4 | ics, Lotus Development Corporation, and Apollo Computer. While believing in the value of software, he did not |
| 68 | 2104 | `' software'` | latin_word | 35/35 | 79.4 | and Apollo Computer. While believing in the value of software, he did not believe in the value of software |
| 69 | 2120 | `' missed'` | latin_word | 35/35 | 80.5 | in the value of software separate from hardware, and missed the opportunity to fund Lotus 1-2- |
| 70 | 2042 | `','` | punct_or_symbol | 35/35 | 86.3 | icomputers at the company of his own founding, Digital Equipment Corporation. At the time the book was |
| 71 | 2122 | `' opportunity'` | latin_word | 35/35 | 79.5 | value of software separate from hardware, and missed the opportunity to fund Lotus 1-2-3 or |
| 72 | 2093 | `','` | punct_or_symbol | 35/35 | 78.9 | , Wang Laboratories, Symbolics, Lotus Development Corporation, and Apollo Computer. While believing in the value of |
| 73 | 2059 | `','` | punct_or_symbol | 35/35 | 79.6 | time the book was published by two computer journal writers, Ken Olsen was competing with other Massachusetts computing companies such |
| 74 | 2045 | `' Corporation'` | latin_word | 35/35 | 116.7 | at the company of his own founding, Digital Equipment Corporation. At the time the book was published by two |
| 75 | 1957 | `'.'` | punct_or_symbol | 35/35 | 84.0 | : The company produced several titles with small development teams. This proved fatal with the rising standards of full priced |
| 76 | 1808 | `' the'` | latin_word | 35/35 | 105.8 | acquire the assets of the largest cable television operator at the time, AT&T Broadband, for US$ |
| 77 | 1945 | `'.\n'` | punct_or_symbol | 35/35 | 85.8 | , which is today known as the Comcast Media Center.\nDocument: The company produced several titles with small development |
| 78 | 2107 | `' did'` | latin_word | 35/35 | 85.4 | . While believing in the value of software, he did not believe in the value of software separate from hardware |
| 79 | 2108 | `' not'` | latin_word | 35/35 | 86.1 | While believing in the value of software, he did not believe in the value of software separate from hardware, |
| 80 | 2026 | `' Ken'` | latin_word | 35/35 | 108.2 | and Digital Equipment Corporation, chronicles the experiences of Ken Olsen racing to design minicomputers at the |

## Example 17

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2097 | `').\n'` | punct_or_symbol | 35/35 | 1.0 | novel The Da Vinci Code (2006).\n |
| 2 | 2086 | `' the'` | latin_word | 35/35 | 3.7 | (2004), and the adaptation of the novel The Da Vinci Code (2006 |
| 3 | 1841 | `'2'` | number | 35/35 | 8.1 | 2015) and The Last Jedi (2017). In 2019 |
| 4 | 2096 | `'6'` | number | 35/35 | 4.7 | the novel The Da Vinci Code (2006).\n |
| 5 | 2083 | `' the'` | latin_word | 35/35 | 5.8 | 3), Wimbledon (2004), and the adaptation of the novel The Da Vinci Code (2 |
| 6 | 2092 | `' ('` | punct_or_symbol | 35/35 | 5.8 | and the adaptation of the novel The Da Vinci Code (2006).\n |
| 7 | 1648 | `' the'` | latin_word | 35/35 | 8.0 | 8 Stewart played King Claudius in Hamlet on the West End and won a second Olivier Award.\nDocument |
| 8 | 2095 | `'0'` | number | 35/35 | 7.5 | of the novel The Da Vinci Code (2006).\n |
| 9 | 2093 | `'2'` | number | 35/35 | 9.7 | the adaptation of the novel The Da Vinci Code (2006).\n |
| 10 | 2088 | `' The'` | latin_word | 35/35 | 10.6 | 004), and the adaptation of the novel The Da Vinci Code (2006).\n |
| 11 | 2094 | `'0'` | number | 35/35 | 13.1 | adaptation of the novel The Da Vinci Code (2006).\n |
| 12 | 2081 | `'),'` | punct_or_symbol | 35/35 | 12.7 | 003), Wimbledon (2004), and the adaptation of the novel The Da Vinci Code |
| 13 | 2090 | `' Vinci'` | latin_word | 35/35 | 14.9 | 4), and the adaptation of the novel The Da Vinci Code (2006).\n |
| 14 | 2087 | `' novel'` | latin_word | 35/35 | 15.1 | 2004), and the adaptation of the novel The Da Vinci Code (2006).\n |
| 15 | 1679 | `'9'` | number | 35/35 | 24.3 | is (born April 20, 1964) is an English actor and film director |
| 16 | 2091 | `' Code'` | latin_word | 35/35 | 17.8 | ), and the adaptation of the novel The Da Vinci Code (2006).\n |
| 17 | 2059 | `' the'` | latin_word | 35/35 | 18.2 | 1), Master and Commander: The Far Side of the World (2003), Dogville ( |
| 18 | 2084 | `' adaptation'` | latin_word | 35/35 | 18.8 | ), Wimbledon (2004), and the adaptation of the novel The Da Vinci Code (20 |
| 19 | 1994 | `' in'` | latin_word | 35/35 | 28.0 | came to the attention of mainstream audiences when he appeared in the British film Gangster No. 1 ( |
| 20 | 2082 | `' and'` | latin_word | 35/35 | 21.1 | 03), Wimbledon (2004), and the adaptation of the novel The Da Vinci Code ( |
| 21 | 2074 | `'),'` | punct_or_symbol | 35/35 | 20.7 | 03), Dogville (2003), Wimbledon (2004), and the adaptation |
| 22 | 2075 | `' Wimbledon'` | latin_word | 35/35 | 24.2 | 3), Dogville (2003), Wimbledon (2004), and the adaptation of |
| 23 | 2085 | `' of'` | latin_word | 35/35 | 24.9 | Wimbledon (2004), and the adaptation of the novel The Da Vinci Code (200 |
| 24 | 1995 | `' the'` | latin_word | 35/35 | 27.3 | to the attention of mainstream audiences when he appeared in the British film Gangster No. 1 (2 |
| 25 | 2066 | `'),'` | punct_or_symbol | 35/35 | 26.2 | Far Side of the World (2003), Dogville (2003), Wimbledon ( |
| 26 | 2089 | `' Da'` | latin_word | 35/35 | 26.2 | 04), and the adaptation of the novel The Da Vinci Code (2006).\n |
| 27 | 2050 | `'),'` | punct_or_symbol | 35/35 | 29.0 | , including A Beautiful Mind (2001), Master and Commander: The Far Side of the World |
| 28 | 1842 | `'0'` | number | 35/35 | 38.3 | 015) and The Last Jedi (2017). In 2019, |
| 29 | 2039 | `' films'` | latin_word | 35/35 | 34.3 | has gone on to appear in a wide variety of films, including A Beautiful Mind (2001 |
| 30 | 2040 | `','` | punct_or_symbol | 35/35 | 30.9 | gone on to appear in a wide variety of films, including A Beautiful Mind (2001), |
| 31 | 2076 | `' ('` | punct_or_symbol | 35/35 | 33.2 | ), Dogville (2003), Wimbledon (2004), and the adaptation of the |
| 32 | 2077 | `'2'` | number | 35/35 | 36.4 | Dogville (2003), Wimbledon (2004), and the adaptation of the novel |
| 33 | 2072 | `'0'` | number | 35/35 | 42.3 | 2003), Dogville (2003), Wimbledon (2004), and |
| 34 | 2027 | `').'` | punct_or_symbol | 35/35 | 36.5 | film A Knight's Tale (2001). He has gone on to appear in a wide variety |
| 35 | 2069 | `' ('` | punct_or_symbol | 35/35 | 38.2 | the World (2003), Dogville (2003), Wimbledon (200 |
| 36 | 1981 | `').'` | punct_or_symbol | 35/35 | 46.0 | Captain America: Civil War (2016). He first came to the attention of mainstream audiences when |
| 37 | 1757 | `' e'` | latin_word | 35/35 | 45.8 | (2012), King Kong in the eponymous 2005 film, Caesar |
| 38 | 2061 | `' ('` | punct_or_symbol | 35/35 | 42.8 | Master and Commander: The Far Side of the World (2003), Dogville (20 |
| 39 | 2068 | `'ville'` | latin_word | 35/35 | 44.1 | of the World (2003), Dogville (2003), Wimbledon (20 |
| 40 | 2054 | `':'` | punct_or_symbol | 35/35 | 44.3 | Mind (2001), Master and Commander: The Far Side of the World (200 |
| 41 | 424 | `'9'` | number | 35/35 | 49.7 | 1895 -- 6 February 1952) was King of the United Kingdom and |
| 42 | 2080 | `'4'` | number | 35/35 | 46.8 | 2003), Wimbledon (2004), and the adaptation of the novel The Da Vinci |
| 43 | 2078 | `'0'` | number | 35/35 | 49.9 | ville (2003), Wimbledon (2004), and the adaptation of the novel The |
| 44 | 2070 | `'2'` | number | 35/35 | 52.3 | World (2003), Dogville (2003), Wimbledon (2004 |
| 45 | 2079 | `'0'` | number | 35/35 | 52.9 | (2003), Wimbledon (2004), and the adaptation of the novel The Da |
| 46 | 2055 | `' The'` | latin_word | 35/35 | 53.0 | (2001), Master and Commander: The Far Side of the World (2003 |
| 47 | 2035 | `' a'` | latin_word | 35/35 | 52.9 | 01). He has gone on to appear in a wide variety of films, including A Beautiful Mind ( |
| 48 | 2034 | `' in'` | latin_word | 35/35 | 59.5 | 001). He has gone on to appear in a wide variety of films, including A Beautiful Mind |
| 49 | 1986 | `' the'` | latin_word | 35/35 | 55.3 | (2016). He first came to the attention of mainstream audiences when he appeared in the British |
| 50 | 2067 | `' Dog'` | latin_word | 35/35 | 56.9 | Side of the World (2003), Dogville (2003), Wimbledon (2 |
| 51 | 2052 | `' and'` | latin_word | 35/35 | 55.8 | A Beautiful Mind (2001), Master and Commander: The Far Side of the World (2 |
| 52 | 2045 | `' ('` | punct_or_symbol | 35/35 | 56.6 | a wide variety of films, including A Beautiful Mind (2001), Master and Commander: The |
| 53 | 1427 | `'The'` | latin_word | 35/35 | 72.9 | At the 83rd Academy Awards, "The King's Speech" won the Academy Award for Best |
| 54 | 2071 | `'0'` | number | 35/35 | 61.4 | (2003), Dogville (2003), Wimbledon (2004), |
| 55 | 1913 | `' the'` | latin_word | 35/35 | 62.3 | .A.R.V.I.S. and the Vision in the Marvel Cinematic Universe, specifically the films Iron Man |
| 56 | 2073 | `'3'` | number | 35/35 | 60.5 | 003), Dogville (2003), Wimbledon (2004), and the |
| 57 | 1682 | `')'` | punct_or_symbol | 35/35 | 64.9 | April 20, 1964) is an English actor and film director. He is |
| 58 | 1433 | `' the'` | latin_word | 35/35 | 67.5 | Academy Awards, "The King's Speech" won the Academy Award for Best Picture, Best Director (Ho |
| 59 | 2062 | `'2'` | number | 35/35 | 65.3 | and Commander: The Far Side of the World (2003), Dogville (200 |
| 60 | 1851 | `'9'` | number | 35/35 | 68.9 | 2017). In 2019, he will play the character of Baloo in |
| 61 | 2048 | `'0'` | number | 35/35 | 70.0 | of films, including A Beautiful Mind (2001), Master and Commander: The Far Side of |
| 62 | 2010 | `' and'` | latin_word | 35/35 | 67.1 | No. 1 (2000), and director Brian Helgeland's film A Knight's |
| 63 | 1963 | `'tron'` | latin_word | 35/35 | 77.4 | 2013), Avengers: Age of Ultron (2015), and Captain America: |
| 64 | 1893 | `'.'` | punct_or_symbol | 35/35 | 70.6 | 1971) is an English actor. He is known for his voice role as J.A |
| 65 | 600 | `' United'` | latin_word | 35/35 | 91.3 | 1952) was King of the United Kingdom and the Dominions of the British Commonwealth from |
| 66 | 2060 | `' World'` | latin_word | 35/35 | 71.7 | ), Master and Commander: The Far Side of the World (2003), Dogville (2 |
| 67 | 2041 | `' including'` | latin_word | 35/35 | 77.7 | on to appear in a wide variety of films, including A Beautiful Mind (2001), Master |
| 68 | 1831 | `'2'` | number | 35/35 | 83.5 | Wars sequel trilogy films, The Force Awakens (2015) and The Last Jedi (2 |
| 69 | 2037 | `' variety'` | latin_word | 35/35 | 76.6 | ). He has gone on to appear in a wide variety of films, including A Beautiful Mind (20 |
| 70 | 2065 | `'3'` | number | 35/35 | 76.3 | The Far Side of the World (2003), Dogville (2003), Wimbledon |
| 71 | 2009 | `'),'` | punct_or_symbol | 35/35 | 80.0 | ster No. 1 (2000), and director Brian Helgeland's film A Knight |
| 72 | 2029 | `' has'` | latin_word | 35/35 | 78.3 | Knight's Tale (2001). He has gone on to appear in a wide variety of films |
| 73 | 2064 | `'0'` | number | 35/35 | 81.7 | : The Far Side of the World (2003), Dogville (2003), |
| 74 | 1891 | `' English'` | latin_word | 35/35 | 105.8 | 7 May 1971) is an English actor. He is known for his voice role as |
| 75 | 2063 | `'0'` | number | 35/35 | 83.0 | Commander: The Far Side of the World (2003), Dogville (2003 |
| 76 | 2028 | `' He'` | latin_word | 35/35 | 84.6 | A Knight's Tale (2001). He has gone on to appear in a wide variety of |
| 77 | 2058 | `' of'` | latin_word | 35/35 | 85.7 | 01), Master and Commander: The Far Side of the World (2003), Dogville |
| 78 | 1880 | `'2'` | number | 35/35 | 96.5 | gli.\nDocument: Paul Bettany (born 27 May 1971) is an |
| 79 | 2057 | `' Side'` | latin_word | 35/35 | 86.1 | 001), Master and Commander: The Far Side of the World (2003), Dog |
| 80 | 1885 | `'9'` | number | 35/35 | 98.1 | Bettany (born 27 May 1971) is an English actor. He is |

## Example 18

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 1545 | `'.\n'` | punct_or_symbol | 35/35 | 1.0 | "Father Brown" with Alec Guinness and Peter Finch.\n |
| 2 | 1542 | `' and'` | latin_word | 35/35 | 2.5 | 54 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 3 | 221 | `' '` | space | 35/35 | 5.5 | 06 - 04) Running time 96 minutes Country United States Language English Budget $ |
| 4 | 1544 | `' Finch'` | latin_word | 35/35 | 4.3 | film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 5 | 1507 | `' the'` | latin_word | 35/35 | 4.4 | and Gertrude Michael. It is based on the Father Brown story "The Blue Cross" by G |
| 6 | 1528 | `' the'` | latin_word | 35/35 | 6.3 | .K. Chesterton, a story which also informed the 1954 film "Father Brown" |
| 7 | 1538 | `'"'` | punct_or_symbol | 35/35 | 7.7 | the 1954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 8 | 1473 | `' a'` | latin_word | 35/35 | 10.1 | and Frank.\nDocument: Father Brown, Detective is a 1934 American mystery film directed by |
| 9 | 1541 | `' Guinness'` | latin_word | 35/35 | 9.8 | 954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 10 | 1543 | `' Peter'` | latin_word | 35/35 | 10.0 | 4 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 11 | 1455 | `' the'` | latin_word | 35/35 | 10.7 | an American tap dancer. He was a member of the Condos Brothers, with siblings Nick and Frank.\n |
| 12 | 1539 | `' with'` | latin_word | 35/35 | 12.1 | 1954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 13 | 1540 | `' Alec'` | latin_word | 35/35 | 13.8 | 1954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 14 | 1522 | `','` | punct_or_symbol | 35/35 | 14.2 | The Blue Cross" by G.K. Chesterton, a story which also informed the 195 |
| 15 | 1529 | `' '` | space | 35/35 | 16.8 | . Chesterton, a story which also informed the 1954 film "Father Brown" with |
| 16 | 1502 | `'.'` | punct_or_symbol | 35/35 | 17.1 | olly, Paul Lukas and Gertrude Michael. It is based on the Father Brown story "The |
| 17 | 1452 | `' a'` | latin_word | 35/35 | 21.8 | 0) was an American tap dancer. He was a member of the Condos Brothers, with siblings Nick |
| 18 | 490 | `' United'` | latin_word | 35/35 | 19.7 | 34th Street (originally released in the United Kingdom as The Big Heart) is a 1 |
| 19 | 1535 | `' "'` | punct_or_symbol | 35/35 | 18.2 | which also informed the 1954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 20 | 1534 | `' film'` | latin_word | 35/35 | 20.8 | story which also informed the 1954 film "Father Brown" with Alec Guinness and Peter Finch |
| 21 | 1275 | `'2'` | number | 35/35 | 30.1 | United Kingdom as Young Bruce Lee) is a 2010 Hong Kong biographical martial arts drama |
| 22 | 1537 | `' Brown'` | latin_word | 35/35 | 21.4 | informed the 1954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 23 | 1515 | `'"'` | punct_or_symbol | 35/35 | 21.0 | based on the Father Brown story "The Blue Cross" by G.K. Chesterton, a story which |
| 24 | 1523 | `' a'` | latin_word | 35/35 | 23.5 | Blue Cross" by G.K. Chesterton, a story which also informed the 1954 |
| 25 | 1536 | `'Father'` | latin_word | 35/35 | 24.9 | also informed the 1954 film "Father Brown" with Alec Guinness and Peter Finch.\n |
| 26 | 1527 | `' informed'` | latin_word | 35/35 | 26.1 | G.K. Chesterton, a story which also informed the 1954 film "Father Brown |
| 27 | 1530 | `'1'` | number | 35/35 | 29.3 | Chesterton, a story which also informed the 1954 film "Father Brown" with Alec |
| 28 | 1525 | `' which'` | latin_word | 35/35 | 29.5 | " by G.K. Chesterton, a story which also informed the 1954 film " |
| 29 | 1521 | `'erton'` | latin_word | 35/35 | 30.7 | "The Blue Cross" by G.K. Chesterton, a story which also informed the 19 |
| 30 | 1512 | `'The'` | latin_word | 35/35 | 30.7 | . It is based on the Father Brown story "The Blue Cross" by G.K. Chesterton, |
| 31 | 1493 | `','` | punct_or_symbol | 35/35 | 32.5 | by Edward Sedgwick and starring Walter Connolly, Paul Lukas and Gertrude Michael. It |
| 32 | 1532 | `'5'` | number | 35/35 | 34.7 | , a story which also informed the 1954 film "Father Brown" with Alec Guinness and |
| 33 | 1516 | `' by'` | latin_word | 35/35 | 34.7 | on the Father Brown story "The Blue Cross" by G.K. Chesterton, a story which also |
| 34 | 1524 | `' story'` | latin_word | 35/35 | 35.0 | Cross" by G.K. Chesterton, a story which also informed the 1954 film |
| 35 | 1506 | `' on'` | latin_word | 35/35 | 39.1 | as and Gertrude Michael. It is based on the Father Brown story "The Blue Cross" by |
| 36 | 1511 | `' "'` | punct_or_symbol | 35/35 | 36.4 | Michael. It is based on the Father Brown story "The Blue Cross" by G.K. Chesterton |
| 37 | 1533 | `'4'` | number | 35/35 | 36.7 | a story which also informed the 1954 film "Father Brown" with Alec Guinness and Peter |
| 38 | 1526 | `' also'` | latin_word | 35/35 | 38.7 | by G.K. Chesterton, a story which also informed the 1954 film "Father |
| 39 | 1514 | `' Cross'` | latin_word | 35/35 | 39.8 | is based on the Father Brown story "The Blue Cross" by G.K. Chesterton, a story |
| 40 | 1519 | `'.'` | punct_or_symbol | 35/35 | 41.8 | Brown story "The Blue Cross" by G.K. Chesterton, a story which also informed the |
| 41 | 1488 | `' and'` | latin_word | 35/35 | 43.7 | 4 American mystery film directed by Edward Sedgwick and starring Walter Connolly, Paul Lukas and Ger |
| 42 | 1449 | `'.'` | punct_or_symbol | 35/35 | 44.9 | 1990) was an American tap dancer. He was a member of the Condos Brothers, |
| 43 | 714 | `' the'` | latin_word | 35/35 | 49.2 | to find that the man assigned to play Santa in the annual Macy's Thanksgiving Day Parade (Percy |
| 44 | 1531 | `'9'` | number | 35/35 | 46.0 | erton, a story which also informed the 1954 film "Father Brown" with Alec Guinness |
| 45 | 1459 | `','` | punct_or_symbol | 35/35 | 49.1 | . He was a member of the Condos Brothers, with siblings Nick and Frank.\nDocument: Father Brown |
| 46 | 1505 | `' based'` | latin_word | 35/35 | 48.6 | Lukas and Gertrude Michael. It is based on the Father Brown story "The Blue Cross" |
| 47 | 1465 | `'.\n'` | punct_or_symbol | 35/35 | 50.1 | the Condos Brothers, with siblings Nick and Frank.\nDocument: Father Brown, Detective is a 1 |
| 48 | 1508 | `' Father'` | latin_word | 35/35 | 51.9 | Gertrude Michael. It is based on the Father Brown story "The Blue Cross" by G.K |
| 49 | 1510 | `' story'` | latin_word | 35/35 | 51.3 | ude Michael. It is based on the Father Brown story "The Blue Cross" by G.K. Chest |
| 50 | 1497 | `' and'` | latin_word | 35/35 | 52.2 | wick and starring Walter Connolly, Paul Lukas and Gertrude Michael. It is based on the |
| 51 | 1503 | `' It'` | latin_word | 35/35 | 54.0 | , Paul Lukas and Gertrude Michael. It is based on the Father Brown story "The Blue |
| 52 | 1520 | `' Chest'` | latin_word | 35/35 | 54.7 | story "The Blue Cross" by G.K. Chesterton, a story which also informed the 1 |
| 53 | 1509 | `' Brown'` | latin_word | 35/35 | 55.3 | trude Michael. It is based on the Father Brown story "The Blue Cross" by G.K. |
| 54 | 1517 | `' G'` | latin_word | 35/35 | 57.5 | the Father Brown story "The Blue Cross" by G.K. Chesterton, a story which also informed |
| 55 | 1438 | `' '` | space | 35/35 | 72.9 | 1918September 16, 1990) was an American tap dancer |
| 56 | 1504 | `' is'` | latin_word | 35/35 | 60.5 | Paul Lukas and Gertrude Michael. It is based on the Father Brown story "The Blue Cross |
| 57 | 1470 | `','` | punct_or_symbol | 35/35 | 60.9 | with siblings Nick and Frank.\nDocument: Father Brown, Detective is a 1934 American mystery |
| 58 | 1243 | `'dale'` | latin_word | 35/35 | 68.5 | 2 graduate of Blair High School, have formed Oakdale Pictures, a production company in Reno.\nDocument: |
| 59 | 1518 | `'.K'` | other_text | 35/35 | 64.1 | Father Brown story "The Blue Cross" by G.K. Chesterton, a story which also informed the |
| 60 | 1416 | `'.\n'` | punct_or_symbol | 35/35 | 64.5 | appearance from Lana Wood, the sister of Natalie Wood.\nDocument: Steve Condos (October 12 |
| 61 | 1068 | `','` | punct_or_symbol | 35/35 | 79.4 | Horler, is a jazz musician and her mother, Christine, is a foreign languages teacher. Horler |
| 62 | 1474 | `' '` | space | 35/35 | 70.4 | Frank.\nDocument: Father Brown, Detective is a 1934 American mystery film directed by Edward |
| 63 | 1468 | `' Father'` | latin_word | 35/35 | 73.0 | Brothers, with siblings Nick and Frank.\nDocument: Father Brown, Detective is a 1934 |
| 64 | 1445 | `' an'` | latin_word | 35/35 | 68.2 | 16, 1990) was an American tap dancer. He was a member of the |
| 65 | 1513 | `' Blue'` | latin_word | 35/35 | 70.6 | It is based on the Father Brown story "The Blue Cross" by G.K. Chesterton, a |
| 66 | 1411 | `' the'` | latin_word | 35/35 | 73.6 | , with an early screen appearance from Lana Wood, the sister of Natalie Wood.\nDocument: Steve Condos |
| 67 | 1472 | `' is'` | latin_word | 35/35 | 70.8 | Nick and Frank.\nDocument: Father Brown, Detective is a 1934 American mystery film directed |
| 68 | 1467 | `':'` | punct_or_symbol | 35/35 | 76.4 | dos Brothers, with siblings Nick and Frank.\nDocument: Father Brown, Detective is a 193 |
| 69 | 1483 | `' by'` | latin_word | 35/35 | 75.1 | a 1934 American mystery film directed by Edward Sedgwick and starring Walter Connolly, |
| 70 | 1481 | `' film'` | latin_word | 35/35 | 76.2 | Detective is a 1934 American mystery film directed by Edward Sedgwick and starring Walter Conn |
| 71 | 235 | `','` | punct_or_symbol | 35/35 | 84.6 | Country United States Language English Budget $630,000 Box office $2.7 million |
| 72 | 1369 | `'.\n'` | punct_or_symbol | 35/35 | 79.0 | in his teenage years to part of his adult years.\nDocument: Five Finger Exercise: The film stars Ros |
| 73 | 1208 | `' He'` | latin_word | 35/35 | 97.3 | the score.\nDocument: Dana MacDuff: He and his older brother, Brandon R. MacD |
| 74 | 1489 | `' starring'` | latin_word | 35/35 | 80.6 | American mystery film directed by Edward Sedgwick and starring Walter Connolly, Paul Lukas and Gertr |
| 75 | 1246 | `' a'` | latin_word | 35/35 | 89.4 | Blair High School, have formed Oakdale Pictures, a production company in Reno.\nDocument: Bruce Lee, |
| 76 | 1443 | `')'` | punct_or_symbol | 35/35 | 81.0 | September 16, 1990) was an American tap dancer. He was a member |
| 77 | 1354 | `' the'` | latin_word | 35/35 | 83.7 | as Lee's parents, the film is based on the life of Bruce Lee in his teenage years to part |
| 78 | 1484 | `' Edward'` | latin_word | 35/35 | 85.3 | 1934 American mystery film directed by Edward Sedgwick and starring Walter Connolly, Paul |
| 79 | 1430 | `'9'` | number | 35/35 | 105.5 | Condos (October 12, 1918September 16, 19 |
| 80 | 1478 | `'4'` | number | 35/35 | 92.9 | Father Brown, Detective is a 1934 American mystery film directed by Edward Sedgwick and |

## Example 19

| rank | doc idx | token | category | selected | mean rank | snippet |
|---:|---:|---|---|---:|---:|---|
| 1 | 2149 | `')\n'` | punct_or_symbol | 35/35 | 1.0 | Vote 92.7% (first ballot)\n |
| 2 | 1654 | `' '` | space | 35/35 | 2.8 | league game. He registered 85 hits in 382 at bats, yielding a .2 |
| 3 | 2143 | `'.'` | punct_or_symbol | 35/35 | 2.6 | ed 1973 Vote 92.7% (first ballot)\n |
| 4 | 2148 | `' ballot'` | latin_word | 35/35 | 4.7 | 3 Vote 92.7% (first ballot)\n |
| 5 | 2146 | `' ('` | punct_or_symbol | 35/35 | 5.6 | 973 Vote 92.7% (first ballot)\n |
| 6 | 2125 | `' the'` | latin_word | 35/35 | 6.4 | ) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame Inducted 1 |
| 7 | 2147 | `'first'` | latin_word | 35/35 | 6.9 | 73 Vote 92.7% (first ballot)\n |
| 8 | 1664 | `'2'` | number | 35/35 | 6.9 | 382 at bats, yielding a .223 batting average. He batted left-handed |
| 9 | 2140 | `' '` | space | 35/35 | 10.6 | Fame Inducted 1973 Vote 92.7% (first ballot)\n |
| 10 | 2145 | `'%'` | punct_or_symbol | 35/35 | 10.7 | 1973 Vote 92.7% (first ballot)\n |
| 11 | 2023 | `'9'` | number | 35/35 | 11.9 | 1967, 1969 -- 1972) 2 × |
| 12 | 2134 | `' '` | space | 35/35 | 13.0 | of the National Baseball Hall of Fame Inducted 1973 Vote 92.7 |
| 13 | 2144 | `'7'` | number | 35/35 | 13.2 | 1973 Vote 92.7% (first ballot)\n |
| 14 | 2142 | `'2'` | number | 35/35 | 13.8 | ucted 1973 Vote 92.7% (first ballot)\n |
| 15 | 2119 | `' '` | space | 35/35 | 16.8 | , 1967) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of |
| 16 | 2139 | `' Vote'` | latin_word | 35/35 | 17.4 | of Fame Inducted 1973 Vote 92.7% (first ballot)\n |
| 17 | 2141 | `'9'` | number | 35/35 | 19.4 | Inducted 1973 Vote 92.7% (first ballot)\n |
| 18 | 2135 | `'1'` | number | 35/35 | 22.2 | the National Baseball Hall of Fame Inducted 1973 Vote 92.7% |
| 19 | 2133 | `'ed'` | latin_word | 35/35 | 23.2 | Member of the National Baseball Hall of Fame Inducted 1973 Vote 92. |
| 20 | 2132 | `'uct'` | latin_word | 35/35 | 24.9 | retired Member of the National Baseball Hall of Fame Inducted 1973 Vote 92 |
| 21 | 2131 | `' Ind'` | latin_word | 35/35 | 25.6 | 1 retired Member of the National Baseball Hall of Fame Inducted 1973 Vote 9 |
| 22 | 2127 | `' Baseball'` | latin_word | 35/35 | 26.1 | Pirates # 21 retired Member of the National Baseball Hall of Fame Inducted 197 |
| 23 | 2130 | `' Fame'` | latin_word | 35/35 | 28.6 | 21 retired Member of the National Baseball Hall of Fame Inducted 1973 Vote |
| 24 | 2138 | `'3'` | number | 35/35 | 27.9 | Hall of Fame Inducted 1973 Vote 92.7% (first ballot |
| 25 | 1741 | `' the'` | latin_word | 35/35 | 33.5 | ulation was rife that a new owner might move the Saints out of New Orleans, namely Jacksonville, Florida |
| 26 | 2011 | `'0'` | number | 35/35 | 31.9 | 15 × All - Star (1960 -- 1967, 19 |
| 27 | 2115 | `')'` | punct_or_symbol | 35/35 | 29.8 | 1965, 1967) Pittsburgh Pirates # 21 retired Member of the |
| 28 | 409 | `' an'` | latin_word | 35/35 | 38.9 | Detroit in Guys and Dolls, and twice won an Emmy Award for his portrayal of the character Benson Du |
| 29 | 2123 | `' Member'` | latin_word | 35/35 | 31.5 | 67) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame Inducted |
| 30 | 2128 | `' Hall'` | latin_word | 35/35 | 35.0 | # 21 retired Member of the National Baseball Hall of Fame Inducted 1973 |
| 31 | 2126 | `' National'` | latin_word | 35/35 | 33.7 | Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame Inducted 19 |
| 32 | 2122 | `' retired'` | latin_word | 35/35 | 35.3 | 967) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame Induct |
| 33 | 2112 | `'9'` | number | 35/35 | 40.6 | 4, 1965, 1967) Pittsburgh Pirates # 21 retired |
| 34 | 2137 | `'7'` | number | 35/35 | 36.6 | Baseball Hall of Fame Inducted 1973 Vote 92.7% (first |
| 35 | 2061 | `'1'` | number | 35/35 | 37.2 | (1966) World Series MVP (1971) 12 × Gold Glo |
| 36 | 2118 | `' #'` | punct_or_symbol | 35/35 | 39.9 | 5, 1967) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall |
| 37 | 2136 | `'9'` | number | 35/35 | 39.5 | National Baseball Hall of Fame Inducted 1973 Vote 92.7% ( |
| 38 | 2124 | `' of'` | latin_word | 35/35 | 41.5 | 7) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame Inducted |
| 39 | 2121 | `'1'` | number | 35/35 | 44.3 | 1967) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame Ind |
| 40 | 1649 | `' '` | space | 35/35 | 48.2 | did not play another major league game. He registered 85 hits in 382 at bats |
| 41 | 1259 | `' '` | space | 35/35 | 61.1 | 12 - 31) (aged 38) San Juan, Puerto Rico Batted |
| 42 | 2066 | `' '` | space | 35/35 | 53.3 | ) World Series MVP (1971) 12 × Gold Glove Award (19 |
| 43 | 2120 | `'2'` | number | 35/35 | 45.4 | 1967) Pittsburgh Pirates # 21 retired Member of the National Baseball Hall of Fame |
| 44 | 1658 | `' at'` | latin_word | 35/35 | 48.9 | registered 85 hits in 382 at bats, yielding a .223 batting average |
| 45 | 2129 | `' of'` | latin_word | 35/35 | 46.8 | 21 retired Member of the National Baseball Hall of Fame Inducted 1973 Vote |
| 46 | 2097 | `','` | punct_or_symbol | 35/35 | 48.3 | 4 × NL batting champion (1961, 1964, 196 |
| 47 | 2031 | `' '` | space | 35/35 | 62.2 | 969 -- 1972) 2 × World Series champion (1960 |
| 48 | 2113 | `'6'` | number | 35/35 | 50.7 | , 1965, 1967) Pittsburgh Pirates # 21 retired Member |
| 49 | 1928 | `' the'` | latin_word | 35/35 | 65.6 | 17, 1955, for the Pittsburgh Pirates Last MLB appearance October 3, |
| 50 | 2117 | `' Pirates'` | latin_word | 35/35 | 54.3 | 65, 1967) Pittsburgh Pirates # 21 retired Member of the National Baseball |
| 51 | 2103 | `','` | punct_or_symbol | 35/35 | 55.2 | 1961, 1964, 1965, 196 |
| 52 | 2109 | `','` | punct_or_symbol | 35/35 | 56.8 | 1964, 1965, 1967) Pittsburgh Pirates # |
| 53 | 1097 | `' $'` | punct_or_symbol | 35/35 | 59.9 | that the AFL had a tentative agreement to sell a $100 million stake in the league to Platinum |
| 54 | 1895 | `' ('` | punct_or_symbol | 35/35 | 62.9 | 2 - 12 - 31) (aged 38) San Juan, Puerto Rico |
| 55 | 2116 | `' Pittsburgh'` | latin_word | 35/35 | 62.0 | 965, 1967) Pittsburgh Pirates # 21 retired Member of the National |
| 56 | 451 | `'9'` | number | 35/35 | 65.9 | . He also won a Grammy Award in 1995 for his spoken word performance of an audi |
| 57 | 2085 | `')'` | punct_or_symbol | 35/35 | 64.5 | 1961 -- 1972) 4 × NL batting champion (196 |
| 58 | 2092 | `' ('` | punct_or_symbol | 35/35 | 64.7 | 972) 4 × NL batting champion (1961, 1964 |
| 59 | 80 | `'9'` | number | 35/35 | 82.5 | Benson (born February 12, 1976) is an American model, former stripper |
| 60 | 2090 | `' batting'` | latin_word | 35/35 | 71.0 | 1972) 4 × NL batting champion (1961, 19 |
| 61 | 1945 | `' the'` | latin_word | 35/35 | 76.8 | 3, 1972, for the Pittsburgh Pirates MLB statistics Batting average. 3 |
| 62 | 2086 | `' '` | space | 35/35 | 71.7 | 961 -- 1972) 4 × NL batting champion (1961 |
| 63 | 2032 | `'2'` | number | 35/35 | 96.7 | 69 -- 1972) 2 × World Series champion (1960, |
| 64 | 1315 | `'.'` | punct_or_symbol | 35/35 | 112.6 | , for the Pittsburgh Pirates MLB statistics Batting average. 317 Hits 3,00 |
| 65 | 2111 | `'1'` | number | 35/35 | 72.5 | 64, 1965, 1967) Pittsburgh Pirates # 21 |
| 66 | 1153 | `' the'` | latin_word | 35/35 | 76.1 | to shut down the VooDoo came during the Platinum Equity conference call, leading to speculation that he |
| 67 | 2110 | `' '` | space | 35/35 | 74.9 | 964, 1965, 1967) Pittsburgh Pirates # 2 |
| 68 | 1632 | `' '` | space | 35/35 | 95.4 | . He was traded back to the Yankees after the 1953 season but did not play another |
| 69 | 2089 | `' NL'` | latin_word | 35/35 | 79.8 | -- 1972) 4 × NL batting champion (1961, 1 |
| 70 | 2098 | `' '` | space | 35/35 | 75.8 | × NL batting champion (1961, 1964, 1965 |
| 71 | 1953 | `'.'` | punct_or_symbol | 35/35 | 76.3 | , for the Pittsburgh Pirates MLB statistics Batting average. 317 Hits 3,00 |
| 72 | 2093 | `'1'` | number | 35/35 | 79.4 | 72) 4 × NL batting champion (1961, 1964, |
| 73 | 2106 | `'9'` | number | 35/35 | 92.3 | 1, 1964, 1965, 1967) Pittsburgh |
| 74 | 1864 | `','` | punct_or_symbol | 35/35 | 105.1 | 1934 Barrio San Antón, Carolina, Puerto Rico Died: December 31 |
| 75 | 2104 | `' '` | space | 35/35 | 83.5 | 961, 1964, 1965, 1967 |
| 76 | 2114 | `'7'` | number | 35/35 | 83.3 | 1965, 1967) Pittsburgh Pirates # 21 retired Member of |
| 77 | 1553 | `' the'` | latin_word | 35/35 | 87.2 | -handed outfielder in Major League Baseball who played for the Houston Astros and Anaheim Angels.\nDocument: Loren Babe |
| 78 | 2007 | `' ('` | punct_or_symbol | 35/35 | 87.7 | highlights and awards 15 × All - Star (1960 -- 1967 |
| 79 | 2080 | `' '` | space | 35/35 | 87.1 | Gold Glove Award (1961 -- 1972) 4 × NL batting |
| 80 | 1954 | `' '` | space | 35/35 | 115.7 | for the Pittsburgh Pirates MLB statistics Batting average. 317 Hits 3,000 |

