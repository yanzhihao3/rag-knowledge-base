# -*- coding: utf-8 -*-
"""《大模型RAG实战》知识库评测用例（50 条）。

字段说明：
  - relevant_keywords：关键词命中口径（Top-5 文本中出现任一关键词即算命中）
  - answer_terms：答案块定位口径（书内答案段落里的特征词，用于把“答案所在的块”找出来）
  - expected_points：回答准确率评测时，答案应覆盖的要点
  - difficulty：easy / medium / hard（13 / 25 / 12）

答案块命中规则（严格口径）：用 answer_terms 在知识库全部分块里定位出“答案块集合”，
评测时检查其中是否有块出现在 Top-5 结果里；MRR 取答案块在结果中的最佳排名的倒数。
"""

BENCHMARK_CASES = [
    # ================= 第1章 RAG 基础 =================
    {
        "query": "RAG框架是哪一年由哪家公司首先提出的",
        "relevant_keywords": ["2020", "Facebook", "Meta", "Retrieval-Augmented Generation"],
        "answer_terms": ["2020", "Facebook"],
        "expected_points": ["2020", "Facebook/Meta", "检索增强生成框架"],
        "difficulty": "easy",
    },
    {
        "query": "RAG在推理过程中分为哪两个阶段",
        "relevant_keywords": ["检索", "内容生成", "检索阶段", "生成阶段"],
        "answer_terms": ["在检索阶段", "在内容生成阶段"],
        "expected_points": ["检索", "内容生成"],
        "difficulty": "easy",
    },
    {
        "query": "大模型为什么难以回答训练语料截止日期之后的问题",
        "relevant_keywords": ["知识更新", "预训练", "过时", "训练语料"],
        "answer_terms": ["拼多多的CEO", "过时的答案"],
        "expected_points": ["知识源于预训练", "无法实时更新", "答案过时"],
        "difficulty": "easy",
    },
    {
        "query": "RAG与模型微调相比，更适合什么样的场景",
        "relevant_keywords": ["数据集是动态的", "微调", "重新训练", "动态"],
        "answer_terms": ["QLoRA", "数据集是动态的"],
        "expected_points": ["数据集动态变化", "避免频繁重新训练", "微调成本"],
        "difficulty": "medium",
    },
    {
        "query": "RAG如何提高大模型回答的可解释性",
        "relevant_keywords": ["溯源", "来源", "链接", "可解释"],
        "answer_terms": ["溯源网页"],
        "expected_points": ["给出答案来源", "可追溯文档", "增强可信度"],
        "difficulty": "medium",
    },
    {
        "query": "使用私有数据构建RAG为什么能降低数据泄露风险",
        "relevant_keywords": ["数据泄露", "私有数据", "敏感信息", "个性化"],
        "answer_terms": ["Extracting Training Data", "GPT-2"],
        "expected_points": ["知识来自知识库", "降低诱导输出敏感信息风险", "答案更个性化"],
        "difficulty": "medium",
    },
    {
        "query": "RAG的工作流程包括哪三个主要阶段",
        "relevant_keywords": ["数据准备", "数据召回", "答案生成"],
        "answer_terms": ["数据准备", "数据召回", "答案生成"],
        "expected_points": ["数据准备", "数据召回", "答案生成"],
        "difficulty": "easy",
    },
    {
        "query": "向量存储库适合存储什么类型的数据",
        "relevant_keywords": ["非结构化", "文本", "图像", "音频", "语义相似性"],
        "answer_terms": ["非结构化", "语义相似性搜索"],
        "expected_points": ["非结构化数据", "按语义相似性搜索", "向量嵌入"],
        "difficulty": "medium",
    },
    {
        "query": "图数据库以什么形式存储数据并如何检索",
        "relevant_keywords": ["节点", "边", "图数据库", "知识图谱"],
        "answer_terms": ["节点和边"],
        "expected_points": ["节点和边", "关系搜索", "适合实体类问答"],
        "difficulty": "easy",
    },
    {
        "query": "RAG系统有哪些优点",
        "relevant_keywords": ["幻觉", "可扩展性", "可解释性", "成本效益"],
        "answer_terms": ["成本效益", "可扩展性"],
        "expected_points": ["降低幻觉", "可扩展", "可解释", "成本效益"],
        "difficulty": "medium",
    },
    {
        "query": "为什么说RAG系统极其依赖检索模块的质量",
        "relevant_keywords": ["检索的质量", "覆盖率", "时效性", "金股"],
        "answer_terms": ["金股", "时效性"],
        "expected_points": ["检索质量决定答案质量", "知识库覆盖率", "片段时效性"],
        "difficulty": "medium",
    },
    {
        "query": "RETRO论文证明了什么结论",
        "relevant_keywords": ["RETRO", "1/25", "GPT-3", "Trillions of Tokens"],
        "answer_terms": ["RETRO", "1/25"],
        "expected_points": ["1/25参数量", "达到GPT-3效果", "外挂知识库"],
        "difficulty": "hard",
    },
    {
        "query": "RAG有哪些常见的使用场景",
        "relevant_keywords": ["文档问答", "图谱问答", "工具召回", "示例召回"],
        "answer_terms": ["ChatPDF", "工具召回"],
        "expected_points": ["文档问答", "图谱问答", "工具召回", "示例召回"],
        "difficulty": "medium",
    },
    # ================= 第2章 语言模型基础 =================
    {
        "query": "Transformer模型首次提出是在哪篇论文中",
        "relevant_keywords": ["Attention is all you need", "Transformer"],
        "answer_terms": ["Attention is all you need"],
        "expected_points": ["2017年", "Attention is all you need"],
        "difficulty": "easy",
    },
    {
        "query": "Transformer由哪几大部分构成",
        "relevant_keywords": ["词嵌入", "编码器", "解码器", "解码头"],
        "answer_terms": ["解码头", "输出生成"],
        "expected_points": ["词嵌入", "编码器", "解码器", "输出生成"],
        "difficulty": "easy",
    },
    {
        "query": "词向量模型中CBOW和Skip-gram两种模式分别预测什么",
        "relevant_keywords": ["CBOW", "Skip-gram", "词袋", "跳跃"],
        "answer_terms": ["CBOW", "Skip-gram"],
        "expected_points": ["CBOW预测目标词", "Skip-gram预测上下文"],
        "difficulty": "medium",
    },
    {
        "query": "ELMo如何解决一词多义问题",
        "relevant_keywords": ["ELMo", "双向LSTM", "上下文", "动态"],
        "answer_terms": ["双向LSTM", "词义歧义"],
        "expected_points": ["动态词嵌入", "根据上下文调整", "双向LSTM"],
        "difficulty": "medium",
    },
    {
        "query": "BERT的全称是什么，其结构有什么特点",
        "relevant_keywords": ["Bidirectional", "双向编码器", "BERT", "预训练"],
        "answer_terms": ["Bidirectional Encoder Representations", "联合左侧和右侧"],
        "expected_points": ["双向编码器表征", "联合左右上下文", "预训练"],
        "difficulty": "easy",
    },
    {
        "query": "自动编码器由哪两部分组成，核心目标是什么",
        "relevant_keywords": ["编码器网络", "解码器网络", "重建", "压缩"],
        "answer_terms": ["Hinton", "Salakhutdinov"],
        "expected_points": ["编码器压缩", "解码器重建", "学习有效表示"],
        "difficulty": "easy",
    },
    # ================= 第3章 文本召回模型 =================
    {
        "query": "稠密向量检索模型和稀疏向量检索模型各擅长什么",
        "relevant_keywords": ["语义信息", "关键词信息", "稠密", "稀疏"],
        "answer_terms": ["稠密向量检索模型", "稀疏向量检索模型"],
        "expected_points": ["稠密擅长语义", "稀疏擅长关键词", "可混合使用"],
        "difficulty": "easy",
    },
    {
        "query": "理想的文本向量空间应该具有哪两个特点",
        "relevant_keywords": ["对齐性", "均匀性", "各向同性"],
        "answer_terms": ["对齐性", "均匀性"],
        "expected_points": ["对齐性", "均匀性"],
        "difficulty": "hard",
    },
    {
        "query": "RAG场景下的文本召回通常是对称检索还是非对称检索",
        "relevant_keywords": ["非对称检索", "对称检索", "问题", "答案"],
        "answer_terms": ["非对称检索", "对称检索"],
        "expected_points": ["非对称检索", "问题召回答案", "需要问答对训练"],
        "difficulty": "hard",
    },
    {
        "query": "计算文本向量相似度最常用的两种度量方式是什么",
        "relevant_keywords": ["余弦相似度", "欧氏距离"],
        "answer_terms": ["余弦相似度", "欧氏距离"],
        "expected_points": ["余弦相似度", "欧氏距离", "L2正则化后等价"],
        "difficulty": "easy",
    },
    {
        "query": "SimCSE使用什么训练方式训练文本向量化模型",
        "relevant_keywords": ["对比学习", "批内负样本", "交叉熵", "SimCSE"],
        "answer_terms": ["批内负样本", "交叉熵损失"],
        "expected_points": ["对比学习", "批内负样本", "拉近正样本推开负样本"],
        "difficulty": "medium",
    },
    {
        "query": "什么叫做难负例，加入难负例有什么作用",
        "relevant_keywords": ["难负例", "相似度较高", "训练难度"],
        "answer_terms": ["难负例", "同一篇文章"],
        "expected_points": ["与正例相似度高的负例", "提高训练难度", "提升模型效果"],
        "difficulty": "hard",
    },
    {
        "query": "CoSENT相比SBERT效果更好的原因是什么",
        "relevant_keywords": ["CoSENT", "排序损失", "余弦相似度", "训练和推理"],
        "answer_terms": ["CoSENT", "排序损失"],
        "expected_points": ["排序损失", "训练推理一致", "计算逻辑一致"],
        "difficulty": "hard",
    },
    {
        "query": "SGPT基于什么模型架构，句向量是如何计算的",
        "relevant_keywords": ["SGPT", "解码器", "加权求和"],
        "answer_terms": ["SGPT", "加权求和"],
        "expected_points": ["Transformer解码器", "位置靠后权重越大", "加权求和"],
        "difficulty": "hard",
    },
    {
        "query": "TF-IDF解决了朴素词袋模型的哪两个缺点",
        "relevant_keywords": ["TF-IDF", "词频", "常见词", "逆文件频率"],
        "answer_terms": ["IDF值", "逆文件频率"],
        "expected_points": ["消除长度影响", "降低常见词权重", "IDF"],
        "difficulty": "medium",
    },
    {
        "query": "BM25相比TF-IDF有哪些改进",
        "relevant_keywords": ["BM25", "饱和", "平均长度", "avgdl"],
        "answer_terms": ["avgdl", "饱和"],
        "expected_points": ["词频饱和", "相对文本长度", "超参数k和b"],
        "difficulty": "medium",
    },
    {
        "query": "为什么重排序模型（交叉编码器）只用于二次精排而不是直接召回",
        "relevant_keywords": ["交叉编码器", "两阶段", "计算量", "二次排序"],
        "answer_terms": ["交叉编码器", "两阶段"],
        "expected_points": ["计算量大", "两阶段召回", "提高精度"],
        "difficulty": "medium",
    },
    # ================= 第4章 核心技术与优化 =================
    {
        "query": "RAG场景下为什么要在提示词中设置兜底回答方式",
        "relevant_keywords": ["无法回答", "强行生成", "兜底"],
        "answer_terms": ["我无法回答你的问题", "不要回答其他信息"],
        "expected_points": ["召回不到相关文本", "避免强行生成", "直接回复无法回答"],
        "difficulty": "easy",
    },
    {
        "query": "Lost in the Middle论文证明了什么结论",
        "relevant_keywords": ["Lost in the Middle", "中间位置", "正确率"],
        "answer_terms": ["Lost in the Middle"],
        "expected_points": ["中间位置利用差", "答案放开头结尾", "影响回答正确率"],
        "difficulty": "hard",
    },
    {
        "query": "文本切块过长会带来什么问题",
        "relevant_keywords": ["信息损失", "召回准确率", "细节"],
        "answer_terms": ["召回准确率降低", "细节的语义信息"],
        "expected_points": ["向量化信息损失", "召回准确率降低", "限制文本块数"],
        "difficulty": "medium",
    },
    {
        "query": "LangChain的RecursiveCharacterTextSplitter是如何切块的",
        "relevant_keywords": ["RecursiveCharacterTextSplitter", "chunk_size", "chunk_overlap", "分隔符"],
        "answer_terms": ["RecursiveCharacterTextSplitter", "chunk_overlap"],
        "expected_points": ["逐级分隔符", "chunk_size/chunk_overlap", "保留上下文"],
        "difficulty": "hard",
    },
    {
        "query": "向量数据库通常集成了哪些高效向量检索算法",
        "relevant_keywords": ["HNSW", "倒排索引", "乘积量化", "分层可导航"],
        "answer_terms": ["分层可导航小世界", "乘积量化"],
        "expected_points": ["HNSW", "倒排文件索引", "乘积量化"],
        "difficulty": "medium",
    },
    {
        "query": "Faiss的精确检索有什么特点",
        "relevant_keywords": ["精确检索", "暴力", "连续的内存", "准确性"],
        "answer_terms": ["暴力检索", "连续的内存空间"],
        "expected_points": ["暴力计算", "速度慢内存大", "保证准确"],
        "difficulty": "medium",
    },
    {
        "query": "乘积量化是如何降低向量存储成本的",
        "relevant_keywords": ["乘积量化", "切片", "聚类", "量化向量"],
        "answer_terms": ["m_split"],
        "expected_points": ["向量切片", "每段聚类", "用聚类ID表示", "节省内存"],
        "difficulty": "hard",
    },
    {
        "query": "短文本全局信息增强常用的三种方式是什么",
        "relevant_keywords": ["前三句话", "标题", "关键词", "全局信息"],
        "answer_terms": ["TopicRank", "前三句话"],
        "expected_points": ["长文本前三句", "标题", "关键词"],
        "difficulty": "hard",
    },
    {
        "query": "父文本检索解决了召回和输入大模型内容之间的什么矛盾",
        "relevant_keywords": ["父文本", "子文本", "上下文", "ParentDocumentRetriever"],
        "answer_terms": ["ParentDocumentRetriever", "父文本"],
        "expected_points": ["向量化模型要短文本", "LLM要长文本", "召回子文本输入父文本"],
        "difficulty": "medium",
    },
    {
        "query": "HyDE方法的核心思路是什么",
        "relevant_keywords": ["HyDE", "假答案", "融合查询向量", "对称检索"],
        "answer_terms": ["HyDE", "假答案"],
        "expected_points": ["生成假答案", "取平均", "转为对称检索"],
        "difficulty": "hard",
    },
    {
        "query": "多轮对话场景中为什么要先对历史对话进行总结再召回",
        "relevant_keywords": ["历史对话", "总结", "关键信息", "缺"],
        "answer_terms": ["北京有哪些好玩景点", "总结句"],
        "expected_points": ["当前问题缺关键信息", "总结历史", "提升召回效果"],
        "difficulty": "medium",
    },
    {
        "query": "RRF倒数排序融合算法是如何给候选文本打分的",
        "relevant_keywords": ["倒数排序融合", "RRF", "60", "排序位置"],
        "answer_terms": ["EnsembleRetriever", "倒数排序融合"],
        "expected_points": ["按排名打分", "k通常为60", "融合后重新排序"],
        "difficulty": "medium",
    },
    {
        "query": "召回环节效果评估常用的两个指标是什么",
        "relevant_keywords": ["命中率", "平均倒数排名", "MRR"],
        "answer_terms": ["平均倒数排名", "MRR"],
        "expected_points": ["命中率", "平均倒数排名MRR"],
        "difficulty": "medium",
    },
    {
        "query": "RAGAS评估框架主要关注哪三个方面",
        "relevant_keywords": ["忠实度", "答案相关性", "上下文相关性", "RAGAS"],
        "answer_terms": ["忠实度", "上下文相关性"],
        "expected_points": ["忠实度", "答案相关性", "上下文相关性"],
        "difficulty": "medium",
    },
    {
        "query": "LoRA参数高效微调的原理是什么",
        "relevant_keywords": ["LoRA", "低秩分解", "参数高效", "全连接"],
        "answer_terms": ["LoRA", "低秩分解"],
        "expected_points": ["低秩分解", "只训练A和B", "大幅节省显存"],
        "difficulty": "medium",
    },
    # ================= 第5章 RAG范式演变 =================
    {
        "query": "基础RAG系统在检索方面存在哪些问题",
        "relevant_keywords": ["语义歧义", "粒度不匹配", "稀疏检索", "局部相似性"],
        "answer_terms": ["语义歧义", "粒度不匹配"],
        "expected_points": ["语义歧义", "粒度不匹配", "全局与局部相似性"],
        "difficulty": "medium",
    },
    {
        "query": "先进RAG系统的优化措施可以归纳为哪几个方面",
        "relevant_keywords": ["数据清理", "微调嵌入", "增强检索", "构建提示词"],
        "answer_terms": ["数据清理", "微调嵌入", "增强检索"],
        "expected_points": ["数据清理", "微调嵌入", "增强检索", "构建提示词"],
        "difficulty": "medium",
    },
    {
        "query": "大模型主导的RAG系统中Agent是如何工作的",
        "relevant_keywords": ["Agent", "工具", "检索-读取-检索", "推理引擎"],
        "answer_terms": ["检索-读取-检索", "AgentFinish"],
        "expected_points": ["LLM是推理引擎", "决定何时调用工具", "检索-读取-检索"],
        "difficulty": "hard",
    },
    # ================= 第7/8章 LangChain与实战 =================
    {
        "query": "LangChain的两大核心概念是什么",
        "relevant_keywords": ["组件", "链", "LangChain"],
        "answer_terms": ["两个核心概念"],
        "expected_points": ["组件", "链"],
        "difficulty": "easy",
    },
    {
        "query": "基于GPT-4V的多模态RAG系统如何处理PDF文档",
        "relevant_keywords": ["多模态", "GPT-4V", "Qdrant", "图片", "文本"],
        "answer_terms": ["GPT-4V", "Qdrant"],
        "expected_points": ["图片文本分离", "多模态向量存储", "统一向量空间生成"],
        "difficulty": "medium",
    },

    # ================= 表格类问题（答案在【图片型表格】里）=================
    # 这一组是 2026-09 补的：原来的 50 题全是正文概念题，测不出表格/代码能力
    # （所有 answer_terms 都能在正文里找到）。加上这两组后，评测集才能反映
    # "面向开发者的技术文档助手"这个场景——它要回答"表里有什么"和"代码怎么写"。
    {
        "query": "表1-1 里 RAG 和模型微调各自的挑战是什么",
        "relevant_keywords": ["表1-1", "挑战", "召回系统", "过拟合"],
        "answer_terms": ["可靠的召回系统", "过拟合"],
        "expected_points": ["RAG需要可靠的召回系统", "微调在数据不足时有风险"],
        "difficulty": "medium",
    },
    {
        "query": "表1-2 里 RETRO 模型的参数量和知识库大小分别是多少",
        "relevant_keywords": ["RETRO", "70亿", "2万亿"],
        "answer_terms": ["RETRO", "2万亿"],
        "expected_points": ["70亿参数", "2万亿知识库"],
        "difficulty": "easy",
    },
    {
        "query": "共现概率矩阵里，单词i和k相关时比值的取值有什么特点",
        "relevant_keywords": ["共现概率", "趋近于1", "相关"],
        "answer_terms": ["趋近于1"],
        "expected_points": ["趋近于1", "相关时很大或不相关时很小"],
        "difficulty": "medium",
    },
    {
        "query": "表2-2 里 LLaMA 预训练数据中 CommonCrawl 占比多少",
        "relevant_keywords": ["CommonCrawl", "67.0%", "数据集"],
        "answer_terms": ["CommonCrawl", "67.0%"],
        "expected_points": ["67.0%", "3.3TB"],
        "difficulty": "easy",
    },
    {
        "query": "表4-1 里 LangChain 提供了哪些特殊文本切块方法",
        "relevant_keywords": ["特殊文本切块", "Markdown", "LaTeX", "HTML"],
        "answer_terms": ["MarkdownTextSplitter", "HTMLHeaderTextSplitter"],
        "expected_points": ["PythonCodeTextSplitter", "MarkdownTextSplitter",
                            "LatexTextSplitter", "HTMLHeaderTextSplitter"],
        "difficulty": "medium",
    },
    {
        "query": "表4-2 里 METRIC_INNER_PRODUCT 对应哪种距离",
        "relevant_keywords": ["METRIC_INNER_PRODUCT", "内积", "距离"],
        "answer_terms": ["METRIC_INNER_PRODUCT"],
        "expected_points": ["内积"],
        "difficulty": "easy",
    },
    {
        "query": "表4-3 里混合精度训练时 FP32 二阶矩占多少显存",
        "relevant_keywords": ["FP32", "二阶矩", "显存"],
        "answer_terms": ["FP32二阶矩", "24GB"],
        "expected_points": ["24GB"],
        "difficulty": "medium",
    },
    {
        "query": "表4-4 里 RAGAS 评测的“是否支持”指标输入和输出是什么",
        "relevant_keywords": ["是否支持", "完全支持", "RAGAS"],
        "answer_terms": ["是否支持", "完全支持"],
        "expected_points": ["输入是x、d、y", "输出是完全支持/部分支持/不支持"],
        "difficulty": "medium",
    },
    {
        "query": "表7-1 里 LangChain 的 SystemMessage 有什么作用",
        "relevant_keywords": ["SystemMessage", "系统消息", "模型如何表现"],
        "answer_terms": ["SystemMessage", "规定模型如何表现"],
        "expected_points": ["系统消息", "规定模型如何表现"],
        "difficulty": "easy",
    },
    {
        "query": "表7-2 里哪个召回器的作用是小块文本向量当作索引、返回大块文本",
        "relevant_keywords": ["召回器", "小块文本", "大块文本"],
        "answer_terms": ["ParentDocumentRetriever"],
        "expected_points": ["ParentDocumentRetriever", "小块检索大块返回"],
        "difficulty": "medium",
    },

    # ================= 代码类问题（答案在【图片型代码】里）=================
    {
        "query": "LangChain 里怎么用 NLTK 做文本切分",
        "relevant_keywords": ["NLTK", "NLTKTextSplitter", "文本切分"],
        "answer_terms": ["NLTKTextSplitter"],
        "expected_points": ["NLTKTextSplitter", "split_text"],
        "difficulty": "easy",
    },
    {
        "query": "怎么用 Faiss 建立向量库并做相似度检索",
        "relevant_keywords": ["FAISS", "相似度", "检索"],
        "answer_terms": ["similarity_search_with_score"],
        "expected_points": ["FAISS.from_texts", "similarity_search_with_score"],
        "difficulty": "medium",
    },
    {
        "query": "召回内容上下文扩充的代码怎么写",
        "relevant_keywords": ["ParentDocumentRetriever", "parent_splitter", "child_splitter"],
        "answer_terms": ["ParentDocumentRetriever"],
        "expected_points": ["ParentDocumentRetriever", "parent_splitter", "child_splitter"],
        "difficulty": "medium",
    },
    {
        "query": "LangChain 怎么实现 HyDE 方法",
        "relevant_keywords": ["HyDE", "HypotheticalDocumentEmbedder"],
        "answer_terms": ["HypotheticalDocumentEmbedder"],
        "expected_points": ["HypotheticalDocumentEmbedder"],
        "difficulty": "medium",
    },
    {
        "query": "CBOW 模型的网络结构代码怎么写",
        "relevant_keywords": ["CBOW", "nn.Embedding", "词向量"],
        "answer_terms": ["nn.Embedding"],
        "expected_points": ["嵌入层", "线性层", "前向传播"],
        "difficulty": "medium",
    },
    {
        "query": "CBOW 训练时优化器选择的是什么",
        "relevant_keywords": ["优化器", "SGD", "随机梯度下降"],
        "answer_terms": ["torch.optim.SGD"],
        "expected_points": ["随机梯度下降", "SGD"],
        "difficulty": "easy",
    },
    {
        "query": "位置编码的代码怎么实现",
        "relevant_keywords": ["位置编码", "getPositionEncoding"],
        "answer_terms": ["getPositionEncoding"],
        "expected_points": ["位置编码矩阵", "正弦函数"],
        "difficulty": "medium",
    },
    {
        "query": "怎么自定义 Agent 的输出解析器",
        "relevant_keywords": ["CustomOutputParser", "AgentFinish", "解析"],
        "answer_terms": ["CustomOutputParser"],
        "expected_points": ["自定义解析器类", "AgentFinish", "Final Answer"],
        "difficulty": "hard",
    },
    {
        "query": "怎么用代码把 PDF 里的图片提取出来",
        "relevant_keywords": ["load_pdf", "fitz", "图片"],
        "answer_terms": ["load_pdf"],
        "expected_points": ["PyMuPDF/fitz", "逐页提取图片", "保存元数据"],
        "difficulty": "hard",
    },
    {
        "query": "提示词模板的示例代码怎么写",
        "relevant_keywords": ["question_answering", "prompt", "示例代码"],
        "answer_terms": ["question_answering"],
        "expected_points": ["prompt模板", "上下文与问题占位符"],
        "difficulty": "medium",
    },
]


# ============================================================
# 用例分类（按场景配比）
# ============================================================
# 前 50 条是正文概念题（原本就有）；后 20 条是 2026-09 补的"图片类"题：
#   索引 50-59 = 表格题（答案在图片型表格里）
#   索引 60-69 = 代码题（答案在图片型代码里）
# 为什么按索引划分而不是给每条加字段：改动小、一眼能看出顺序，但**新增用例必须追加在末尾**。
_TABLE_RANGE = range(50, 60)
_CODE_RANGE = range(60, 70)


def case_category(index: int) -> str:
    """返回用例类别：text / table / code。"""
    if index in _TABLE_RANGE:
        return "table"
    if index in _CODE_RANGE:
        return "code"
    return "text"
