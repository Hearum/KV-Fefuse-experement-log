# GLM Judge Review Samples

该文件抽查 MuSiQue GLM rerun 中原始 judge 与 GLM judge 不一致的样本，用于判断 GLM 是否变宽或误判。


## full_rate1

- false_to_true: 18
- true_to_false: 0

### False -> True

#### False -> True sample 1

- Main Question: The Unwinding author volunteered for which organisation?
- Sub Question: Which organization did George Packer volunteer for?
- Ground Truth: Peace Corps
- Predicted clean: Peace Corps
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何关于乔治·帕克志愿加入的组织信息，因此未能回答问题，与标准答案“Peace Corps”不一致。
- GLM Correct: True | GLM Reason: 预测答案“Peace Corps”与标准答案完全一致，准确包含了问题所问及的关键信息，语义上完全等价，正确地回答了George Packer所志愿服务的组织。

#### False -> True sample 2

- Main Question: What year was the father of Katherine of England crowned?
- Sub Question: What year was King Henry III of England crowned?
- Ground Truth: Henry III was first crowned in 1216 and had a second coronation in 1220.
- Predicted clean: 1216     Answer: 1216
- Old Correct: False | Old Reason: 预测答案“1216”仅包含了标准答案的部分信息。标准答案明确指出亨利三世在1216年首次加冕，并在1220年举行了第二次加冕仪式。预测答案遗漏了1220年这一关键信息，未能完整且准确地回答问题。
- GLM Correct: True | GLM Reason: 预测答案“1216”准确包含了标准答案中关于国王亨利三世首次加冕年份的关键信息。虽然标准答案还提到了第二次加冕的年份1220，但预测答案正确回答了问题所问的加冕时间，符合核心语义要求。

#### False -> True sample 3

- Main Question: What is the place of birth of the person who submitted the first version of the equal rights amendment to congress in 1923?
- Sub Question: Who submitted the first version of the Equal Rights Amendment to Congress in 1923?
- Ground Truth: Alice Paul, the head of the National Women's Party, revised the proposed Equal Rights Amendment in 1923.
- Predicted clean: Alice Paul
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何实质性的内容。因此，它未能包含标准答案中的关键信息（Alice Paul），也没有回答出问题所问的内容。
- GLM Correct: True | GLM Reason: 预测答案“Alice Paul”准确包含了标准答案中的核心关键信息。问题询问的是在1923年向国会提交第一版《平等权利修正案》的人物，预测答案正确指出了该人物为Alice Paul，与标准答案在语义上完全一致，符合正确回答的标准。

#### False -> True sample 4

- Main Question: What city is located in the county containing Qiantong?
- Sub Question: What city is located in Ninghai County?
- Ground Truth: Qiantong is a town located in Ninghai County.
- Predicted clean: Qiantong
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何信息来回答问题。标准答案明确指出前童是位于宁海县的一个镇，而预测答案未能包含这一关键信息，因此无法被视为正确回答了问题。
- GLM Correct: True | GLM Reason: 预测答案“Qiantong”直接提取了标准答案中的核心地点信息，准确回答了关于宁海县所在城市/城镇的问题，与标准答案在语义上保持一致，符合正确回答的标准。

#### False -> True sample 5

- Main Question: What pantheon is the God of the underworld in ancient Egypt a part of?
- Sub Question: Who is the god of the underworld in ancient Egyptian mythology?
- Ground Truth: Osiris is the god of the underworld in ancient Egyptian mythology.
- Predicted clean: Osiris
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何实质性的内容或关键信息，因此未能回答关于古埃及神话中冥界之神是谁的问题。
- GLM Correct: True | GLM Reason: 预测答案“Osiris”准确给出了古埃及神话中的冥界之神的名字，包含了标准答案的核心关键信息。虽然它没有使用完整的句子结构，但在语义上与标准答案完全等价，正确地回答了问题。

#### False -> True sample 6

- Main Question: What province shares a border with the province where Lago District is located?
- Sub Question: In which province is Lago District located?
- Ground Truth: Niassa Province
- Predicted clean: Niassa Province
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何关于 Lago District 所在省份的信息，因此未能回答问题，也不包含标准答案中的关键信息。
- GLM Correct: True | GLM Reason: 预测答案与标准答案完全一致，均为“Niassa Province”，准确包含了标准答案的关键信息，语义上完全等价，正确回答了Lago District所在的省份。

### True -> False

无。


## online_qk_rate015

- false_to_true: 18
- true_to_false: 1

### False -> True

#### False -> True sample 1

- Main Question: The Unwinding author volunteered for which organisation?
- Sub Question: Which organization did George Packer volunteer for?
- Ground Truth: Peace Corps
- Predicted clean: Peace Corps
- Old Correct: False | Old Reason: 预测答案部分为空，没有提供任何关于George Packer志愿加入的组织信息，因此未能包含标准答案“Peace Corps”的关键信息，无法回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案“Peace Corps”与标准答案完全一致，准确包含了问题所要求的关键信息，语义上完全等价，正确地回答了George Packer所志愿服务的组织。

#### False -> True sample 2

- Main Question: When did the crowning of the person under whom the Chapter House was built happen?
- Sub Question: When was Henry III crowned?
- Ground Truth: Henry III was first crowned in 1216 at Gloucester Cathedral, and then again at Westminster Abbey in 1220.
- Predicted clean: 17 May 1220   Henry III was crowned on 17 May 1220 at Westminster Abbey after his initial coronation in 1216 at Gloucester Cathedral was deemed improper by the Pope.
- Old Correct: False | Old Reason: 预测答案“17 May 1220”仅包含了亨利三世第二次加冕的具体日期，遗漏了标准答案中极其关键的首次加冕时间（1216年）。由于未能完整回答问题所涉及的全部加冕时间点，因此判定为错误。
- GLM Correct: True | GLM Reason: 预测答案不仅包含了标准答案中的关键时间点（1216年和1220年），还准确提及了对应的加冕地点（Gloucester Cathedral和Westminster Abbey）。虽然预测答案补充了具体的日期（17 May 1220）和教皇认为初次加冕不当的背景信息，但其核心语义与标准答案完全一致，正确且全面地回答了问题。

#### False -> True sample 3

- Main Question: Where did the author of The National Dream go to university?
- Sub Question: Who is the author of The National Dream?
- Ground Truth: Pierre Berton
- Predicted clean: The author of *The National Dream* is Pierre Berton.
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何关于《The National Dream》作者的信息，因此未能回答问题。
- GLM Correct: True | GLM Reason: 预测答案明确指出了《The National Dream》的作者是Pierre Berton，包含了标准答案的所有关键信息，且语义完全一致，准确回答了问题。

#### False -> True sample 4

- Main Question: What is the record label of the Metalworks Studio founder?
- Sub Question: Who is the founder of Metalworks Studio?
- Ground Truth: Gil Moore
- Predicted clean: Gil Moore
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何具体的人名或信息，因此未能包含标准答案中的关键信息“Gil Moore”，无法回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案“Gil Moore”与标准答案完全一致，准确包含了问题所要求的关键信息，语义上完全等价，正确地回答了关于Metalworks Studio创始人的问题。

#### False -> True sample 5

- Main Question: The city which contains Broadhurst Park shares a border with what town?
- Sub Question: Which towns share a border with the city of Manchester, England?
- Ground Truth: Moston shares a border with the city of Manchester.
- Predicted clean: Middleton and Moston
- Old Correct: False | Old Reason: 预测答案虽然包含了标准答案中的关键信息“Moston”，但它额外添加了“Middleton”。根据判断标准，预测答案必须与标准答案在语义上等价，不能包含标准答案中未提及的多余信息。因此，预测答案与标准答案不一致，判定为错误。
- GLM Correct: True | GLM Reason: 预测答案中包含了标准答案的关键信息“Moston”，并且直接回答了问题中关于哪些城镇与曼彻斯特接壤的询问。虽然预测答案额外补充了“Middleton”，但并未遗漏标准答案的核心内容，符合包含关键信息即视为正确的评估标准。

#### False -> True sample 6

- Main Question: What company succeeded the owner of Empire Sports Network?
- Sub Question: What company succeeded Adelphia Communications Corporation?
- Ground Truth: Time Warner Cable acquired assets from the bankrupt Adelphia, and a new company, Windjammer Communications, was formed from the sale of some of these systems.
- Predicted clean: Windjammer Cable succeeded Adelphia Communications Corporation.
- Old Correct: False | Old Reason: 预测答案不完整且存在偏差。标准答案指出，Adelphia的资产被时代华纳有线收购，同时部分系统出售后成立了Windjammer Communications。预测答案仅提及Windjammer Cable（名称与标准答案略有出入），遗漏了时代华纳有线这一关键主体，未能全面准确地回答问题。
- GLM Correct: True | GLM Reason: 预测答案指出了Windjammer Cable继承了Adelphia Communications Corporation，这与标准答案中提到的“a new company, Windjammer Communications, was formed from the sale of some of these systems”在语义上是一致的，包含了关键信息，因此判断为正确。

### True -> False

#### True -> False sample 1

- Main Question: Bancroft's county borders what county?
- Sub Question: Which county is Bancroft, Ontario located in?
- Ground Truth: Bancroft, Ontario is located in Hastings County.
- Predicted clean: North Hastings
- Old Correct: True | Old Reason: 预测答案“North Hastings”指的是“Hastings County”的北部区域。Bancroft 在行政区划上完全属于 Hastings County，且常被称为 North Hastings 的中心。因此，该预测答案在地理和行政语义上与标准答案等价，正确回答了问题。
- GLM Correct: False | GLM Reason: 预测答案“North Hastings”指的是一个地区，而问题询问的是具体的县。标准答案明确指出Bancroft位于Hastings County，预测答案未包含此关键信息，且语义上不等价。


## online_draft_rate015

- false_to_true: 16
- true_to_false: 1

### False -> True

#### False -> True sample 1

- Main Question: The Unwinding author volunteered for which organisation?
- Sub Question: Which organization did George Packer volunteer for?
- Ground Truth: Peace Corps
- Predicted clean: Peace Corps
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于George Packer所志愿服务的组织信息，因此未能包含标准答案中的关键信息，无法回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案“Peace Corps”与标准答案完全一致，准确包含了问题所要求的关键信息，语义上完全等价，正确地回答了George Packer所志愿服务的组织。

#### False -> True sample 2

- Main Question: What year was the father of Katherine of England crowned?
- Sub Question: What year was King Henry III of England crowned?
- Ground Truth: Henry III was first crowned in 1216 and had a second coronation in 1220.
- Predicted clean: 1216     Answer: 1216
- Old Correct: False | Old Reason: 预测答案“1216”仅包含了标准答案中的部分信息。标准答案明确指出亨利三世在1216年首次加冕，并在1220年举行了第二次加冕仪式。预测答案遗漏了1220年这一关键信息，未能完整、准确地回答问题。
- GLM Correct: True | GLM Reason: 预测答案“1216”准确包含了标准答案中关于国王亨利三世首次加冕年份的关键信息。虽然标准答案还提及了1220年的第二次加冕，但预测答案正确回答了问题所问的加冕年份，语义上与标准答案的核心内容保持一致。

#### False -> True sample 3

- Main Question: What is the record label for the singer who originally sang You Belong To Me.
- Sub Question: What record label was Joni James signed to when she released 'You Belong To Me' in 1952?
- Ground Truth: MGM Records
- Predicted clean: MGM Records
- Old Correct: False | Old Reason: 您在提示词中给出了“预测答案:”的字样，但并未在其后提供任何实际的预测文本内容。由于缺乏需要评估的预测答案，无法将其与标准答案“MGM Records”进行比对，因此无法判断其正确性。
- GLM Correct: True | GLM Reason: 预测答案“MGM Records”与标准答案完全一致，准确地包含了问题所询问的关于Joni James在1952年发行歌曲时的唱片公司这一关键信息，语义上完全等价，因此判断为正确。

#### False -> True sample 4

- Main Question: When did the person who first brought a postal service into Umayyad lands become caliph?
- Sub Question: Who first introduced the postal service in the Umayyad Caliphate?
- Ground Truth: Mu'awiyah introduced the postal service in the Umayyad Caliphate.
- Predicted clean: Mu'awiyah introduced the postal service in the Umayyad Caliphate.
- Old Correct: False | Old Reason: 预测答案部分为空，没有提供任何实质性的内容或人名信息，因此未能包含标准答案中的关键信息，也没有正确回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案与标准答案完全一致，准确包含了“Mu'awiyah”这一关键人物信息，并且语义上完全等价，正确地回答了谁在倭马亚哈里发国引入了邮政服务的问题。

#### False -> True sample 5

- Main Question: What city is located in the county containing Qiantong?
- Sub Question: What city is located in Ninghai County?
- Ground Truth: Qiantong is a town located in Ninghai County.
- Predicted clean: Qiantong
- Old Correct: False | Old Reason: 预测答案“Qiantong”存在事实性错误。根据标准答案，Qiantong（前童）是一个镇，而不是城市。此外，预测答案仅提供了一个名称，未能准确回答问题中关于“位于宁海县的哪个城市”的询问，与标准答案在语义上不等价。
- GLM Correct: True | GLM Reason: 预测答案“Qiantong”准确提取了标准答案中的核心实体信息。虽然标准答案补充了它是位于宁海县的一个镇，但预测答案已经包含了回答该问题所需的关键地点名称，在语义上与标准答案的核心意图保持一致。

#### False -> True sample 6

- Main Question: What province shares a border with the province where Lago District is located?
- Sub Question: In which province is Lago District located?
- Ground Truth: Niassa Province
- Predicted clean: Lago District is located in Niassa Province.
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于 Lago District 所在省份的信息，因此未能回答问题，也不包含标准答案中的关键信息。
- GLM Correct: True | GLM Reason: 预测答案明确指出了Lago District位于Niassa Province，包含了标准答案中的核心地理信息。两者在语义上完全等价，准确且直接地回答了问题，符合判断标准。

### True -> False

#### True -> False sample 1

- Main Question: Who is the child of Sigrid Eskilsdotter's child?
- Sub Question: Who are the children of Christina Gyllenstierna?
- Ground Truth: Christina Gyllenstierna's children include a daughter (mother of King Gustav Vasa) and a child who was the parent of Sigrid Sture.
- Predicted clean: Sigrid Sture and King Gustav Vasa of Sweden.
- Old Correct: True | Old Reason: 预测答案中提到的“Sigrid Sture”和“King Gustav Vasa”与标准答案中描述的 Christina Gyllenstierna 的后代完全对应。虽然标准答案表述为她子女的孩子，但预测答案准确指出了这些关键后代人物，在核心语义上与标准答案保持一致。
- GLM Correct: False | GLM Reason: 预测答案直接将“Sigrid Sture”和“King Gustav Vasa”列为Christina Gyllenstierna的子女，这与标准答案的语义不符。标准答案明确指出这两人是她的孙辈，即她的女儿是Gustav Vasa的母亲，另一个孩子是Sigrid Sture的父母之一。因此，预测答案在核心亲属关系上存在事实性错误。


## offline10_hybrid_old70_docgen30_draft005

- false_to_true: 12
- true_to_false: 2

### False -> True

#### False -> True sample 1

- Main Question: What administrative territorial entity is the owner of Ciudad Deportiva located?
- Sub Question: What administrative territorial entity owns the Ciudad Deportiva sports complex in Nuevo Laredo, Mexico?
- Ground Truth: The Ciudad Deportiva sports complex is owned by the Nuevo Laredo Municipality.
- Predicted clean: Nuevo Laredo Municipality
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于该体育设施所属行政领土实体的信息，未能回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案“Nuevo Laredo Municipality”直接提取了标准答案的核心实体信息，准确回答了问题所问的行政领土实体。两者在语义上完全等价，符合评估标准。

#### False -> True sample 2

- Main Question: What is the record label of the Metalworks Studio founder?
- Sub Question: Who is the founder of Metalworks Studio?
- Ground Truth: Gil Moore
- Predicted clean: The founder of Metalworks Studio is Gil Moore.
- Old Correct: False | Old Reason: 预测答案部分为空，没有提供任何实质性的回答内容。虽然您在提示中给出了“Gil Moore”这一信息，但作为待评估的预测答案文本本身是缺失的，因此无法判定其正确回答了问题。
- GLM Correct: True | GLM Reason: 预测答案明确指出了Metalworks Studio的创始人是Gil Moore，包含了标准答案中的所有关键信息，且在语义上与标准答案完全等价，准确回答了问题。

#### False -> True sample 3

- Main Question: What company succeeded the owner of Empire Sports Network?
- Sub Question: What company succeeded Adelphia Communications Corporation?
- Ground Truth: Time Warner Cable acquired assets from the bankrupt Adelphia, and a new company, Windjammer Communications, was formed from the sale of some of these systems.
- Predicted clean: Windjammer Cable succeeded Adelphia Communications Corporation.
- Old Correct: False | Old Reason: 预测答案仅提到了Windjammer Cable，但遗漏了标准答案中的另一个关键主体Time Warner Cable。标准答案指出Time Warner Cable也收购了Adelphia的资产，预测答案未能完整包含这一关键信息，因此判定为错误。
- GLM Correct: True | GLM Reason: 预测答案指出了Windjammer Cable继承了Adelphia Communications Corporation，这与标准答案中提到的新公司Windjammer Communications由Adelphia出售的部分系统组建而成这一关键信息在语义上是一致的，正确回答了问题。

#### False -> True sample 4

- Main Question: The 1967 Federation Cup is an instance of this tournament which is named after what organization?
- Sub Question: Which organization is the Federation Cup in women's tennis named after?
- Ground Truth: The International Tennis Federation (ITF)
- Predicted clean: International Tennis Federation (ITF)
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于联合会杯命名来源的信息，因此未能回答问题。
- GLM Correct: True | GLM Reason: 预测答案“International Tennis Federation (ITF)”与标准答案“The International Tennis Federation (ITF)”在语义上完全等价，仅缺少了一个定冠词“The”，属于措辞上的细微差异，包含了所有关键信息，正确回答了问题。

#### False -> True sample 5

- Main Question: Who is the child of the Victim of Romance performer?
- Sub Question: Who performed the song 'Victim of Romance'?
- Ground Truth: Michelle Phillips
- Predicted clean: Michelle Phillips
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何关于歌曲演唱者的信息，因此未能回答问题，与标准答案“Michelle Phillips”不符。
- GLM Correct: True | GLM Reason: 预测答案“Michelle Phillips”与标准答案完全一致，准确地给出了演唱歌曲“Victim of Romance”的表演者，包含了标准答案的关键信息，语义完全等价，因此判断为正确。

#### False -> True sample 6

- Main Question: Who is the father of the performer of Je dis aime?
- Sub Question: Who is the father of Matthieu Chedid?
- Ground Truth: Louis Chedid
- Predicted clean: Louis Chedid
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何关于 Matthieu Chedid 父亲的信息，因此未能包含标准答案中的关键信息，也没有在语义上与标准答案保持一致。
- GLM Correct: True | GLM Reason: 预测答案“Louis Chedid”与标准答案完全一致，准确地给出了Matthieu Chedid父亲的名字，包含了所有关键信息且语义等价，因此判断为正确。

### True -> False

#### True -> False sample 1

- Main Question: When was John's eldest son crowned?
- Sub Question: When was John's eldest son crowned?
- Ground Truth: John I of Cyprus (John II of Jerusalem) was crowned on May 11, 1284.
- Predicted clean: 1284     Answer: May 11, 1284
- Old Correct: True | Old Reason: 预测答案“1284”准确给出了约翰长子加冕的年份，与标准答案中的时间信息完全一致。虽然预测答案省略了具体的月和日，但已经正确回答了问题所询问的核心时间点，符合语义等价的判断标准。
- GLM Correct: False | GLM Reason: 预测答案仅给出了年份“1284”，而标准答案明确指出了约翰长子加冕的具体日期为“May 11, 1284”。预测答案缺失了月份和具体日期这一关键信息，未能完整且准确地回答问题。

#### True -> False sample 2

- Main Question: Where is Ulrich Walter's employer headquartered?
- Sub Question: Who is Ulrich Walter's employer?
- Ground Truth: Ulrich Walter's employers were the German astronaut team and the German Aerospace Center (DLR).
- Predicted clean: German astronaut team
- Old Correct: True | Old Reason: 预测答案“German astronaut team”是标准答案“German astronaut team and the German Aerospace Center (DLR)”的一部分。它准确包含了标准答案中的关键信息之一，符合允许部分匹配的评估标准，因此判定为正确回答了问题。
- GLM Correct: False | GLM Reason: 预测答案只包含了标准答案中的部分信息（German astronaut team），遗漏了另一个关键雇主“German Aerospace Center (DLR)”。由于没有完整包含标准答案的所有关键信息，因此判断为错误。


## draft32b_smart_top2_mean_global

- false_to_true: 10
- true_to_false: 2

### False -> True

#### False -> True sample 1

- Main Question: When did the person who first brought a postal service into Umayyad lands become caliph?
- Sub Question: Who first introduced the postal service in the Umayyad Caliphate?
- Ground Truth: Mu'awiyah introduced the postal service in the Umayyad Caliphate.
- Predicted clean: Mu'awiyah introduced the postal service in the Umayyad Caliphate.
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何实质性的内容或文字信息，因此未能包含标准答案中的关键信息（Mu'awiyah），无法正确回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案与标准答案完全一致，准确包含了“Mu'awiyah”这一关键人物信息，且语义上完全等价，正确回答了谁在倭马亚哈里发国引入邮政服务的问题。

#### False -> True sample 2

- Main Question: What company succeeded the owner of Empire Sports Network?
- Sub Question: Who owned Empire Sports Network?
- Ground Truth: Empire Sports Network was owned by the Adelphia Communications Corporation.
- Predicted clean: Adelphia Communications Corporation
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于 Empire Sports Network 所有者的信息，未能包含标准答案中的关键信息，因此无法正确回答所提出的问题。
- GLM Correct: True | GLM Reason: 预测答案“Adelphia Communications Corporation”直接给出了问题所问的拥有者，包含了标准答案中的核心实体信息，与标准答案在语义上完全等价，准确回答了问题。

#### False -> True sample 3

- Main Question: What is the mouth of watercourse for the body of water where Bartram's Covered Bridge is located?
- Sub Question: What watercourse does Crum Creek flow into?
- Ground Truth: Crum Creek flows into the Delaware River.
- Predicted clean: Delaware River
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于Crum Creek汇入哪条水系的信息，因此未能回答问题。
- GLM Correct: True | GLM Reason: 预测答案“Delaware River”直接给出了问题所问的水系名称，包含了标准答案的核心关键信息，且在语义上与标准答案完全等价，准确回答了问题。

#### False -> True sample 4

- Main Question: The 1967 Federation Cup is an instance of this tournament which is named after what organization?
- Sub Question: Which organization is the Federation Cup in women's tennis named after?
- Ground Truth: The International Tennis Federation (ITF)
- Predicted clean: International Tennis Federation (ITF)
- Old Correct: False | Old Reason: 预测答案为空，没有提供任何关于联合会杯命名来源的信息，因此未能回答问题。
- GLM Correct: True | GLM Reason: 预测答案“International Tennis Federation (ITF)”与标准答案“The International Tennis Federation (ITF)”在语义上完全等价，包含了所有关键信息，仅缺少了定冠词“The”，属于允许的措辞细微差异，正确回答了问题。

#### False -> True sample 5

- Main Question: Who is the child of the Victim of Romance performer?
- Sub Question: Who performed the song 'Victim of Romance'?
- Ground Truth: Michelle Phillips
- Predicted clean: Michelle Phillips
- Old Correct: False | Old Reason: 预测答案部分为空，没有提供任何关于歌曲演唱者的信息，因此未能包含标准答案中的关键信息，无法正确回答该问题。
- GLM Correct: True | GLM Reason: 预测答案“Michelle Phillips”与标准答案完全一致，准确地给出了演唱歌曲“Victim of Romance”的歌手信息，包含了标准答案的所有关键信息，语义完全等价，因此判断为正确。

#### False -> True sample 6

- Main Question: In which county did Snappy Tomato Pizza form?
- Sub Question: Which county is Fort Mitchell, Kentucky located in?
- Ground Truth: Kenton County
- Predicted clean: Kenton County
- Old Correct: False | Old Reason: 预测答案部分为空白，没有提供任何关于Fort Mitchell, Kentucky所在县的文字信息，因此未能回答问题，与标准答案“Kenton County”不符。
- GLM Correct: True | GLM Reason: 预测答案“Kenton County”与标准答案完全一致，准确地给出了Fort Mitchell, Kentucky所在的县名，包含了标准答案的所有关键信息，语义完全等价，因此判断为正确。

### True -> False

#### True -> False sample 1

- Main Question: When was John's eldest son crowned?
- Sub Question: When was John's eldest son crowned?
- Ground Truth: John I of Cyprus (John II of Jerusalem) was crowned on May 11, 1284.
- Predicted clean: 1284     Answer: March 24, 1284
- Old Correct: True | Old Reason: 预测答案“1284”准确包含了标准答案中的核心年份信息。虽然它省略了具体的月和日，但针对“When”这一提问，提供正确的年份已足以回答该问题，在语义上与标准答案的核心要点保持一致。
- GLM Correct: False | GLM Reason: 预测答案仅给出了年份“1284”，而标准答案明确指出加冕日期为“May 11, 1284”。预测答案缺失了关键的月份和日期信息，未能完整包含标准答案的核心内容，因此判断为错误。

#### True -> False sample 2

- Main Question: Where is Ulrich Walter's employer headquartered?
- Sub Question: Who is Ulrich Walter's employer?
- Ground Truth: Ulrich Walter's employers were the German astronaut team and the German Aerospace Center (DLR).
- Predicted clean: German astronaut team
- Old Correct: True | Old Reason: 预测答案“German astronaut team”是标准答案中明确提及的雇主之一。虽然它省略了另一个雇主“German Aerospace Center (DLR)”，但已经准确包含了标准答案的核心关键信息，正确回答了问题。
- GLM Correct: False | GLM Reason: 预测答案仅包含了标准答案中的部分信息，遗漏了另一个重要的雇主“German Aerospace Center (DLR)”。由于没有完整包含标准答案的关键信息，因此判断为错误。
