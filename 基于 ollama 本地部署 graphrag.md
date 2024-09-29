# 基于 ollama 本地部署 graphrag

本测试基于 ollama 本地的大模型，使用 graphrag 服务，对西游记的文本构建 knowledge graph 并完成 RAG. 

参考文档：

- https://github.com/echonoshy/cgft-llm/tree/master/graph-rag#ollama本地部署graph-rag服务
- https://chishengliu.com/zh-tw/posts/graphrag-local-ollama/

这两篇文档已经写的很详细，尤其是本地部署遇到的问题。但参照执行时，第一篇文档的 local search 不能工作，第二篇文档的 global search 不工作。多次尝试后终于解决，特整理如下

<aside>
💡

测试环境：

python: 3.12

graphrag: 0.3.6

ollama: 0.3.12

</aside>

### 一、构建python 环境

1. 创建 conda env: graphrag

```bash
$ conda create -n graphrag python=3.12
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done

$ conda activate graphrag
```

1. 安装必要的 package

```bash
conda activate graphrag
pip install graphrag
pip install ollama
```

1. 创建目录

```bash
$ mkdir -p graphrag
$ cd graphrag
$ mkdir -p xiyouji/input
$ mkdir models
```

1. 下载必要的配置文件

```bash
$ git clone https://github.com/jiyzhang/ollama_graphrag
```

1. 当前的目录结构

```bash
$ tree
.
├── models
├── ollama_graphrag
│   ├── LICENSE
│   ├── README.md
│   ├── input
│   │   └── 霸王茶姬.txt
│   ├── settings.yaml
│   └── src
│       ├── Modelfile
│       ├── embedding.py
│       └── openai_embeddings_llm.py
└── xiyouji
    └── input

5 directories, 7 files
```

### 二、准备大模型

参考: https://github.com/echonoshy/cgft-llm/tree/master/graph-rag

1. 下载模型： Qwen/Qwen2-7B-Instruct-GGUF （qwen2-7b-instruct-q5_k_m.ggu

```bash
# 当前目录为 ~/graphrag
pip install huggingface-hub
# 通过huggingface镜像站下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-7B-Instruct-GGUF qwen2-7b-instruct-q5_k_m.gguf --local-dir ./models/
```

output:

```bash
$ export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-7B-Instruct-GGUF qwen2-7b-instruct-q5_k_m.gguf --local-dir ./models/

Downloading 'qwen2-7b-instruct-q5_k_m.gguf' to 'models/.cache/huggingface/download/qwen2-7b-instruct-q5_k_m.gguf.7aeb11bb9b36d47625bd61514a3ac085826537144b375bb1d66b42738127e425.incomplete' (resume from 10485760/5444828960)
qwen2-7b-instruct-q5_k_m.gguf: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.44G/5.44G [02:18<00:00, 39.4MB/s]
Download complete. Moving file to models/qwen2-7b-instruct-q5_k_m.gguf
models/qwen2-7b-instruct-q5_k_m.gguf
```

 2. 复制Modelfile

```bash
$ cp ollama_graphrag/src/Modelfile ./models
```

1. 编辑和检查 Modelfile

```bash
  1 FROM ~/graphrag/models/qwen2-7b-instruct-q5_k_m.gguf
  2
  3 PARAMETER temperature 1
  4 PARAMETER num_ctx 4096
```

1. 生成 ollama model

```bash
$ cd ~/graphrag/models
$ ls
Modelfile                     qwen2-7b-instruct-q5_k_m.gguf
$ ollama create qwen2-7b-instruct-q5_k_m -f ./Modelfile
transferring model data 100%
using existing layer sha256:7aeb11bb9b36d47625bd61514a3ac085826537144b375bb1d66b42738127e425
using existing layer sha256:77c91b422cc9fce701d401b0ecd74a2d242dafd84983aa13f0766e9e71936db2
using existing layer sha256:75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de
using existing layer sha256:f02dd72bb2423204352eabc5637b44d79d17f109fdb510a7c51455892aa2d216
creating new layer sha256:a0307b5f9f51a1e04b26041d2019a903d88c07cacab76318200aa693400183ab
writing manifest
success
```

1. 检查 ollama模型

```bash
$ ollama list
NAME                           	ID          	SIZE  	MODIFIED
**qwen2-7b-instruct-q5_k_m:latest	d3e11d0a8d19	5.4 GB	8 seconds ago**
qwen2:7b                       	dd314f039b9d	4.4 GB	2 hours ago
nomic-embed-text:latest        	0a109f422b47	274 MB	5 hours ago
llama3.1:latest                	91ab477bec9d	4.7 GB	4 weeks ago
llama3:8b                      	365c0bd3c000	4.7 GB	3 months ago
mistral:latest                 	61e88e884507	4.1 GB	4 months ago
llava:13b                      	0d0eb4d7f485	8.0 GB	5 months ago
starcoder2:7b                  	0679cedc1189	4.0 GB	6 months ago
gemma:7b-instruct              	430ed3535049	5.2 GB	6 months ago
gemma:7b                       	cb9e0badc99d	4.8 GB	6 months ago
```

1. 对基于 ollama 的 embedding 模型做测试

```bash
$ curl http://localhost:11434/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2-7b-instruct-q5_k_m",
        "input": ["why is the sky blue?", "why is the grass green?"]
    }'
{"object":"list","data":[{"object":"embedding","embedding":[0.011606854,0.0042322706,0.010206669,0.003419882,-0.01460967,0.00814337,0.016074961,-0.010905064,0.017151315,0.03496108,-0.0059881313,0.009213811,0.011349993,0.0015921381,0.0066692503,0.015
....
```

### 三、初始化 graphrag

1. 下载西游记文件

```bash
curl https://www.gutenberg.org/cache/epub/23962/pg23962.txt >./xiyouji/input/book.txt
```

```bash
$ curl https://www.gutenberg.org/cache/epub/23962/pg23962.txt >./xiyouji/input/book.txt
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 2211k  100 2211k    0     0   581k      0  0:00:03  0:00:03 --:--:--  581k
```

1. graphrag 目录初始化

> python -m graphrag.index --init --root ./xiyouji
> 

```bash
$ python -m graphrag.index --init --root ./xiyouji
Initializing project at ./xiyouji
⠋ GraphRAG Indexer %
$ tree xiyouji
xiyouji
├── input
│   └── book.txt
├── prompts
│   ├── claim_extraction.txt
│   ├── community_report.txt
│   ├── entity_extraction.txt
│   └── summarize_descriptions.txt
└── settings.yaml

3 directories, 6 files

$ cd xiyouji
$ ls -la
total 24
drwxr-xr-x  6 richardz  staff   192 Sep 12 22:15 .
drwxr-xr-x  3 richardz  staff    96 Sep 12 22:10 ..
-rw-r--r--@ 1 richardz  staff    28 Sep 12 22:15 .env
drwxr-xr-x  3 richardz  staff    96 Sep 12 22:13 input
drwxr-xr-x@ 6 richardz  staff   192 Sep 12 22:15 prompts
-rw-r--r--@ 1 richardz  staff  5329 Sep 12 22:15 settings.yaml
```

1. 修改 settings.yaml, 使用ollama本地部署的大模型和本地 embedding 模型
    
    > 这里使用 **qwen2-7b-instruct-q5_k_m** 做为 LLM 和 embedding 模型
    > 

```bash
# 复制 ollama_graphrag/settings.yaml
cp ollama_graphrag/settings.yaml ./xiyouji
```

### 四、修改 graphrag package 文件

graphrag只支持 openAI 和 Azure OpenAI 环境，并不支持基于 ollama 的本地大模型环境。需要对安装的 package内容做一定的修改才可以使用。具体需要修改的文件如下

```bash
～/miniconda3/envs/grag/lib/python3.12/site-packages/graphrag/llm/openai/openai_embeddings_llm.py
～/miniconda3/envs/grag/lib/python3.12/site-packages/graphrag/query/llm/oai/embedding.py
～/miniconda3/envs/grag/lib/python3.12/site-packages/graphrag/query/structured_search/global_search/search.py
～/miniconda3/envs/grag/lib/python3.12/site-packages/graphrag/query/structured_search/local_search/search.py
```

> 具体目录结构，根据conda 的安装方式略有不同
> 
1. 修改 graphrag/llm/openai/openai_embeddings_llm.py

```bash
$ cp ollama_graphrag/src/openai_embeddings_llm.py ～/miniconda3/envs/grag/lib/python3.12/site-packages/graphrag/llm/openai/openai_embeddings_llm.py
```

1. 修改 graphrag/query/llm/oai/embedding.py

```bash
$ cp ollama_graphrag/src/embedding.py ～/miniconda3/envs/grag/lib/python3.12/site-packages/graphrag/query/llm/oai/embedding.py
```

1. 修改 graphrag/query/structured_search/global_search/search.py

注释掉 209 到 212 行，增加 213行

```bash
209 #            search_messages = [
210 #                {"role": "system", "content": search_prompt},
211 #                {"role": "user", "content": query},
212 #            ]
213             search_messages = [ {"role": "user", "content": search_prompt + "\n\n### USER QUESTION ### \n\n" + query} ]
```

 4.  修改 graphrag/query/structured_search/local_search/search.py

```bash
 78 #            search_messages = [
 79 #                {"role": "system", "content": search_prompt},
 80 #                {"role": "user", "content": query},
 81 #            ]
 82             search_messages = [ {"role": "user", "content": search_prompt + "\n\n### USER QUESTION ### \n\n" + query} ]
```

注释掉 78 到 81 行，增加 82 行

### 五、索引文件并创建 knowledge graph

命令：python -m graphrag.index --root ./xiyouji

> 非常耗时。在 Mac Studio 上，需要18 个小时
> 

```bash
$ python -m graphrag.index --root ./xiyouji
Logging enabled at ～/graphrag/xiyouji/output/20240912-224249/reports/indexing-engine.log

🚀 create_base_text_units
                                   id                                              chunk                          chunk_id                        document_ids  n_tokens
0    1e17281c57c89ae5df4e8cc31a87dfdf  The Project Gutenberg eBook of 西遊記\n    \nThi...  1e17281c57c89ae5df4e8cc31a87dfdf  [6bb959359e54493abf6f13abb302b826]      1200
1    bfb880891057777f987403b4021d721e  \n！萬物資生，乃順承天。」至此，地始凝結。再五千四百歲，正當丑會，重濁下\n凝，有水，有火...  bfb880891057777f987403b4021d721e  [6bb959359e54493abf6f13abb302b826]      1200
2    e8de3e9c8ad6e5bd413c0cdd27f31ce9  �高上帝，駕座金闕雲宮靈霄寶殿，聚集仙卿，見有金光燄燄\n，即命千里眼、順風耳開南天門觀看。...  e8de3e9c8ad6e5bd413c0cdd27f31ce9  [6bb959359e54493abf6f13abb302b826]      1200
3    7f25cea1803c66f593f2db4c1299fb66  出一個石猴，應聲高叫道：「我進去，我進去。」好\n猴！也是他：\n        今日芳名顯...  7f25cea1803c66f593f2db4c1299fb66  [6bb959359e54493abf6f13abb302b826]      1200
4    be87a0ebaa5ef5c5d55cb4f632c625f6  進來。」那些猴有膽大的，都跳進去了\n﹔膽小的，一個個伸頭縮頸，抓耳撓腮，大聲叫喊，纏一會，...  be87a0ebaa5ef5c5d55cb4f632c625f6  [6bb959359e54493abf6f13abb302b826]      1200
..                                ...                                                ...                               ...                                 ...       ...
199  3cb26aabe2111fc02ac274dd0767ef7d  光佛。南無大通光佛。南無才光佛。南無旃檀功德佛。南無\n鬥戰勝佛。南無觀世音菩薩。南無大勢至...  3cb26aabe2111fc02ac274dd0767ef7d  [6bb959359e54493abf6f13abb302b826]      1200
200  a7cff3d70833c1df766fc6e0bb24e39f   be bound by the terms of this agreement. Ther...  a7cff3d70833c1df766fc6e0bb24e39f  [6bb959359e54493abf6f13abb302b826]      1200
201  8d66a18f1e4748e48356f95da9e59ecf   upon request, of the work in its original “Pl...  8d66a18f1e4748e48356f95da9e59ecf  [6bb959359e54493abf6f13abb302b826]      1200
202  9ba4ec10aaef88ae6d5c0233f7211b00   IMPLIED, INCLUDING BUT NOT\nLIMITED TO WARRAN...  9ba4ec10aaef88ae6d5c0233f7211b00  [6bb959359e54493abf6f13abb302b826]      1151
203  781a14005ead87a46e20d29c73c02a59   www.gutenberg.org.\n\nThis website includes i...  781a14005ead87a46e20d29c73c02a59  [6bb959359e54493abf6f13abb302b826]        51

[1021 rows x 5 columns]
🚀 create_base_extracted_entities
                                        entity_graph
0  <graphml xmlns="http://graphml.graphdrawing.or...
🚀 create_summarized_entities
                                        entity_graph
0  <graphml xmlns="http://graphml.graphdrawing.or...
🚀 create_base_entity_graph
   level                                    clustered_graph
0      0  <graphml xmlns="http://graphml.graphdrawing.or...
1      1  <graphml xmlns="http://graphml.graphdrawing.or...
2      2  <graphml xmlns="http://graphml.graphdrawing.or...
3      3  <graphml xmlns="http://graphml.graphdrawing.or...
🚀 create_final_entities
                                   id                                               name          type  ... graph_embedding                                      text_unit_ids                              description_embedding
0    b45241d70f0e43fca764df95b2b81f77                                  PROJECT GUTENBERG  ORGANIZATION  ...            None  [1e17281c57c89ae5df4e8cc31a87dfdf, 781a14005ea...  [-0.994062066078186, -1.0057852268218994, 0.96...
1    4119fd06010c494caa07f439b333f4c5                                                西遊記           GEO  ...            None                 [1e17281c57c89ae5df4e8cc31a87dfdf]  [2.5446560382843018, 5.979983806610107, 1.8459...
2    d3835bf3dda84ead99deadbeac5d0d7d                                        CHENG'EN WU        PERSON  ...            None                 [1e17281c57c89ae5df4e8cc31a87dfdf]  [0.7440456748008728, 1.5120842456817627, 2.734...
3    077d2820ae1845bcbb1803379a3d1eae                                  DECEMBER 22, 2007         EVENT  ...            None                 [1e17281c57c89ae5df4e8cc31a87dfdf]  [-1.3623356819152832, 2.447719097137451, -0.07...
4    3671ea0dd4e84c1a9b02c5ab2c8f4bac                                      UNITED STATES           GEO  ...            None                 [1e17281c57c89ae5df4e8cc31a87dfdf]  [0.7594135999679565, 3.4986586570739746, 1.090...
..                                ...                                                ...           ...  ...             ...                                                ...                                                ...
238  303e3c6100514911846813bc27852927                                THE TRADEMARK OWNER  ORGANIZATION  ...            None                 [9ba4ec10aaef88ae6d5c0233f7211b00]  [-0.14460529386997223, -0.9727663397789001, -0...
239  4e961d18211e44a08ad54135b9413e5d            ANY AGENT OR EMPLOYEE OF THE FOUNDATION  ORGANIZATION  ...            None                 [9ba4ec10aaef88ae6d5c0233f7211b00]  [-2.944389581680298, -1.1579360961914062, -0.9...
240  7479419bae8f4ccba9dd665fb0222476  ANYONE PROVIDING COPIES OF PROJECT GUTENBERG™ ...  ORGANIZATION  ...            None                 [9ba4ec10aaef88ae6d5c0233f7211b00]  [-0.4143495261669159, -1.9666807651519775, -1....
241  7842714ddccc4a80a82e55fa6676f4a3  ANY VOLUNTEERS ASSOCIATED WITH THE PRODUCTION,...  ORGANIZATION  ...            None                 [9ba4ec10aaef88ae6d5c0233f7211b00]  [-3.343294620513916, -0.9023085832595825, 1.01...
242  e84f2230f2a74d76bbd83101825b3e02                           EMAIL NEWSLETTER SERVICE  ORGANIZATION  ...            None                 [781a14005ead87a46e20d29c73c02a59]  [-1.7256109714508057, -2.508983612060547, 0.46...

[3164 rows x 8 columns]

🚀 create_final_nodes
       level                                              title          type                                        description                                          source_id  ...   entity_type  community                 top_level_node_id  x
y
0          0                                  PROJECT GUTENBERG  ORGANIZATION  Project Gutenberg is a volunteer-run digital l...  1e17281c57c89ae5df4e8cc31a87dfdf,781a14005ead8...  ...           NaN        NaN  b45241d70f0e43fca764df95b2b81f77  0
0
1          0                                                西遊記           GEO  The geographical location mentioned in this te...                   1e17281c57c89ae5df4e8cc31a87dfdf  ...           GEO        NaN  4119fd06010c494caa07f439b333f4c5
0  0
2          0                                        CHENG'EN WU        PERSON  Wu Cheng'en is the author of the Chinese novel...                   1e17281c57c89ae5df4e8cc31a87dfdf  ...        PERSON        NaN  d3835bf3dda84ead99deadbeac5d0d7d  0
0
3          0                                  DECEMBER 22, 2007         EVENT  The release date mentioned in the text is Dece...                   1e17281c57c89ae5df4e8cc31a87dfdf  ...         EVENT        NaN  077d2820ae1845bcbb1803379a3d1eae  0
0
4          0                                      UNITED STATES           GEO  This text mentions that the eBook can be used ...                   1e17281c57c89ae5df4e8cc31a87dfdf  ...           GEO        NaN  3671ea0dd4e84c1a9b02c5ab2c8f4bac  0
0
...      ...                                                ...           ...                                                ...                                                ...  ...           ...        ...                               ... ..
..
12651      3                                THE TRADEMARK OWNER  ORGANIZATION  The organization being indemnified in the agre...                   9ba4ec10aaef88ae6d5c0233f7211b00  ...  ORGANIZATION        NaN  303e3c6100514911846813bc27852927  0
0
12652      3            ANY AGENT OR EMPLOYEE OF THE FOUNDATION  ORGANIZATION  The organization being indemnified in the agre...                   9ba4ec10aaef88ae6d5c0233f7211b00  ...  ORGANIZATION        NaN  4e961d18211e44a08ad54135b9413e5d  0
0
12653      3  ANYONE PROVIDING COPIES OF PROJECT GUTENBERG™ ...  ORGANIZATION  The organization being indemnified in the agre...                   9ba4ec10aaef88ae6d5c0233f7211b00  ...  ORGANIZATION        NaN  7479419bae8f4ccba9dd665fb0222476  0
0
12654      3  ANY VOLUNTEERS ASSOCIATED WITH THE PRODUCTION,...  ORGANIZATION  The organization being indemnified in the agre...                   9ba4ec10aaef88ae6d5c0233f7211b00  ...  ORGANIZATION        NaN  7842714ddccc4a80a82e55fa6676f4a3  0
0
12655      3                           EMAIL NEWSLETTER SERVICE  ORGANIZATION  Project Gutenberg offers an email newsletter s...                   781a14005ead87a46e20d29c73c02a59  ...           NaN        NaN  e84f2230f2a74d76bbd83101825b3e02  0
0

[12656 rows x 15 columns]

🚀 create_final_communities
      id          title  level raw_community                                   relationship_ids                                      text_unit_ids
0      9    Community 9      0             9  [a7c2777f2f9f454ab770dfdf216f44b6, 97f42316385...  [4e9f606b37317d283573e1a6bdaaabfb,9e7eedbce820...
1     18   Community 18      0            18  [9bf60d81f1774c2d939f15d91289bcfb, 52e85400b94...  [0c05e44917af7686c7f3a88082061dcb,1a095d34cef2...
2     10   Community 10      0            10  [432ec02a9fa34d7fad2e407285282d53, 79dcc354894...  [0726743c48b8935d90b3f2e481af8f65,0c05e44917af...
3     14   Community 14      0            14  [cf14af5b6c0345819f95accff7cc6092, 87e1edbfc0c...  [a903e40bdd8f1f2dbbb808f6b498f4a6,e8de3e9c8ad6...
4     13   Community 13      0            13  [10080de4deb2415f843f068854cf1bd9, 0faf308d5a7...  [0e3b134e369df39efb1b67082bfd8076,10bf50c6b6de...
..   ...            ...    ...           ...                                                ...                                                ...
155  312  Community 312      3           312  [a9aafbf9a0b646acacadc780e8236360, 5afd0ee9247...  [1d060992bcb4fd3a05200ca6dc79166e,7ac23e60dfb1...
156  313  Community 313      3           313  [eef605b4151f45cf969fdbc8dc1f8e57, 2bace414354...  [1d060992bcb4fd3a05200ca6dc79166e,69ee291f5d73...
157  319  Community 319      3           319  [5dbe101ab6044cab837a265a365fbd31, dc7a18d4c76...  [5f17bd3434a811ea5169262f7c77a306,e857ca9e01d4...
158  305  Community 305      3           305  [4ab3c14f05354e28a7ffc5629c1f7061, 5bc7f94f834...  [25dfb44f29a57ea2a5c61eb79e72dd62,5a702ea3e412...
159  308  Community 308      3           308  [e659342116a448f7a01f106eee2fc1a8, 722d2b43243...

[320 rows x 6 columns]
🚀 join_text_units_to_entity_ids
                        text_unit_ids                                         entity_ids                                id
0    1e17281c57c89ae5df4e8cc31a87dfdf  [b45241d70f0e43fca764df95b2b81f77, 4119fd06010...  1e17281c57c89ae5df4e8cc31a87dfdf
1    781a14005ead87a46e20d29c73c02a59  [b45241d70f0e43fca764df95b2b81f77, e1973a7560a...  781a14005ead87a46e20d29c73c02a59
2    8d66a18f1e4748e48356f95da9e59ecf  [b45241d70f0e43fca764df95b2b81f77, e1973a7560a...  8d66a18f1e4748e48356f95da9e59ecf
3    bfb880891057777f987403b4021d721e  [19a7f254a5d64566ab5cc15472df02de, e7ffaee9d31...  bfb880891057777f987403b4021d721e
4    4e9f606b37317d283573e1a6bdaaabfb  [27f9fbe6ad8c4a8b9acee0d3596ed57c, e1fd0e904a5...  4e9f606b37317d283573e1a6bdaaabfb
..                                ...                                                ...                               ...
536  419216f85ca68866155ee72a483c5981  [33b9b5e0f8da40ea89b05c9ab5355bb0, b24cb4da575...  419216f85ca68866155ee72a483c5981
537  6332bf2f38f1136bc4f032feb72ba2b9  [6e920d6be7354ed9b380b3bb7b433dff, 44619429a8e...  6332bf2f38f1136bc4f032feb72ba2b9
538  470adc40165d7d1cf11589c712f00d8b  [9e2a79f2695c4bff86fb77158a7a19c8, cd1b9458b5f...  470adc40165d7d1cf11589c712f00d8b
539  48d78dec02db09ca9dcc831a1b2c5c51  [a7c58d4d092e45439ee43218fb17dc3b, 0bebbddee56...  48d78dec02db09ca9dcc831a1b2c5c51
540  9ba4ec10aaef88ae6d5c0233f7211b00  [b8da3b11c3994caca014b48c06b819fd, 315d4a69693...  9ba4ec10aaef88ae6d5c0233f7211b00

[541 rows x 3 columns]

🚀 create_final_relationships
                 source                                             target  weight                                        description  ... human_readable_id source_degree target_degree  rank
0     PROJECT GUTENBERG                                      UNITED STATES     1.0  This text specifies that the eBook can be used...  ...                 0             4             1     5
1     PROJECT GUTENBERG      PROJECT GUTENBERG LITERARY ARCHIVE FOUNDATION     3.0  The Project Gutenberg Literary Archive Foundat...  ...                 1             4             1     5
2     PROJECT GUTENBERG                 PROJECT GUTENBERG ELECTRONIC WORKS     2.0  Project Gutenberg creates, distributes, and ma...  ...                 2             4             2     6
3     PROJECT GUTENBERG                           EMAIL NEWSLETTER SERVICE     1.0  Email newsletter service provided by Project G...  ...                 3             4             1     5
4                   傲來國                                                美猴王     1.0  The monkey king's actions affected the kingdom...  ...                 4             2            23    25
...                 ...                                                ...     ...                                                ...  ...               ...           ...           ...   ...
2629             1.F.5.  SOME STATES DO NOT ALLOW DISCLAIMERS OF CERTAI...     6.0  Section 1.F.5 relates to the disclaimer of war...  ...              2629             1             1     2
2630     THE FOUNDATION                                THE TRADEMARK OWNER     4.0  The Foundation and trademark owner are related...  ...              2630             4             1     5
2631     THE FOUNDATION            ANY AGENT OR EMPLOYEE OF THE FOUNDATION     6.0  An agent or employee of the Foundation is part...  ...              2631             4             1     5
2632     THE FOUNDATION  ANYONE PROVIDING COPIES OF PROJECT GUTENBERG™ ...     4.0  Individuals providing copies are covered under...  ...              2632             4             1     5
2633     THE FOUNDATION  ANY VOLUNTEERS ASSOCIATED WITH THE PRODUCTION,...     4.0  Volunteers involved in production and distribu...  ...              2633             4             1     5

[2634 rows x 10 columns]
🚀 join_text_units_to_relationship_ids
                                   id                                   relationship_ids
0    1e17281c57c89ae5df4e8cc31a87dfdf
1    8d66a18f1e4748e48356f95da9e59ecf  [adcdfdfdcf3e48599119379c71519d66, 071b4a8c4e8...
2    781a14005ead87a46e20d29c73c02a59
3    4e9f606b37317d283573e1a6bdaaabfb  [a7c2777f2f9f454ab770dfdf216f44b6, 97f42316385...
4    e8de3e9c8ad6e5bd413c0cdd27f31ce9  [9bf60d81f1774c2d939f15d91289bcfb, 52e85400b94...
..                                ...                                                ...
526  470adc40165d7d1cf11589c712f00d8b  [3ad45769e35c458d84fa0da077e470fd, 7dc8ed5e46a...
527  a9a67dce88f88a3028686aadcfda912a
528  cdd872c691cae234973267362d869d46
529  48d78dec02db09ca9dcc831a1b2c5c51  [9d88c79c074b4d3790aa85c9aedf416b, 324496f7766...
530  9ba4ec10aaef88ae6d5c0233f7211b00  [9a09dd9d6f9e492789d886c61586498e, 7ccd8844dd6...

[531 rows x 2 columns]

🚀 create_final_community_reports
    community                                       full_content  level  ...                                           findings                                  full_content_json                                    id
0         305  # 妖兵与孙大圣的冲突\n\n社区围绕着妖兵、孙大圣和阵势三个关键实体展开，其中妖兵作为孙大...      3  ...  [{'summary': '妖兵作为故事中的敌人', 'explanation': '妖兵是...  {\n    "title": "妖兵与孙大圣的冲突",\n    "summary": "...
f9162ed4-dd4f-4cb6-bfe0-9082df065107
1         306  # 孫大聖與葫蘆兒、精細鬼&伶俐蟲\n\n社區圍繞孫大聖、葫蘆兒和精細鬼&伶俐蟲這三個關鍵實...      3  ...  [{'summary': '孫大聖的角色', 'explanation': '孫大聖在社區中...  {\n    "title": "孫大聖與葫蘆兒、精細鬼&伶俐蟲",\n    "summa...
94f0d62f-4791-4966-bd6c-3a8a96225789
2         308  # 天竺國王与給孤布金寺的关联\n\n社区围绕着天竺国王和给孤布金寺两个关键实体，通过关系展...      3  ...  [{'summary': '天竺国王与和尚讨论真公主的位置', 'explanation':...  {\n    "title": "天竺國王与給孤布金寺的关联",\n    "summary...
476cdc81-c41e-4e43-b0e9-dd84021ed04d
3         309  # Community of Relationships Involving Bodhisa...      3  ...  [{'summary': 'Bodhisattvas' Role in Guidance a...  {\n    "title": "Community of Relationships In...  328761ec-994e-423d-8de4-fb41d94b8e71
4         310  # Woodfork Community\n\nThe community revolves...      3  ...  [{'summary': 'Woodfork's Role as a Disciple of...  {\n    "title": "Woodfork Community",\n    "su...  51959d00-a713-4893-9e4f-fc6b513d9671
..        ...                                                ...    ...  ...                                                ...                                                ...                                   ...
226         2  # Journey to the West Community\n\nThe Journey...      0  ...  [{'summary': 'Sun Wukong's Mission', 'explanat...  {\n    "title": "Journey to the West Community...  82c31c83-6636-4a24-a7c4-2a6b3051e248
227         3  # Community of Chinese Mythological Entities\n...      0  ...  [{'summary': 'Central Theme:人参树 Connection', '...  {\n    "title": "Community of Chinese Mytholog...  b2f91266-c202-4f07-bf86-63b39c8946a3
228         4  # Journey to the West Community\n\nThe Journey...      0  ...  [{'summary': 'Key Entities and Their Roles', '...  {\n    "title": "Journey to the West Community...  bf526e4e-9b25-40c5-b86f-5808465c6b54
229         5  # Community of Elders and Monks\n\nThis commun...      0  ...  [{'summary': 'Poetry Exchange Between Elders',...  {\n    "title": "Community of Elders and Monks...  9a7e0339-0366-4511-baf4-836d8f3cea5e
230         6  # Community of Entities with Robes and Monks\n...      0  ...  [{'summary': 'Robes as symbols of status and v...  {\n    "title": "Community of Entities with Ro...  3dbdf081-6a4f-4a07-aeeb-05b5d2e2dc66

[231 rows x 10 columns]
🚀 create_final_text_units
                                    id                                               text  n_tokens                        document_ids                                         entity_ids                                   relationship_ids
0     1e17281c57c89ae5df4e8cc31a87dfdf  The Project Gutenberg eBook of 西遊記\n    \nThi...      1200  [6bb959359e54493abf6f13abb302b826]  [b45241d70f0e43fca764df95b2b81f77, 4119fd06010...
1     e8de3e9c8ad6e5bd413c0cdd27f31ce9  �高上帝，駕座金闕雲宮靈霄寶殿，聚集仙卿，見有金光燄燄\n，即命千里眼、順風耳開南天門觀看。...      1200  [6bb959359e54493abf6f13abb302b826]  [e1fd0e904a53409aada44442f23a51cb, d91a266f766...
[9bf60d81f1774c2d939f15d91289bcfb, 52e85400b94...
2     7f25cea1803c66f593f2db4c1299fb66  出一個石猴，應聲高叫道：「我進去，我進去。」好\n猴！也是他：\n        今日芳名顯...      1200  [6bb959359e54493abf6f13abb302b826]  [4a67211867e5464ba45126315a122a8a, e69dc259edb...
[57826115c2a7416794b51e5ffd528860, 8144775ab08...
3     be87a0ebaa5ef5c5d55cb4f632c625f6  進來。」那些猴有膽大的，都跳進去了\n﹔膽小的，一個個伸頭縮頸，抓耳撓腮，大聲叫喊，纏一會，...      1200  [6bb959359e54493abf6f13abb302b826]  [bf4e255cdac94ccc83a56435a5e4b075, 3b040bcc19f...
[67176aa5d43f4e48a0b453bee0f5de2a, 351c60f2b5c...
4     42fd641d39dc2fd85887c61b97ecf18e  �高叫道：「大王若是這般遠慮，真所\n謂道心開發也。如今五蟲之內，惟有三等名色不伏閻王老子所...      1200  [6bb959359e54493abf6f13abb302b826]  [bf4e255cdac94ccc83a56435a5e4b075, eae4259b19a...
[eb46034a82544e9c9519aee6af2cb078, c8f1c0f638c...
...                                ...                                                ...       ...                                 ...                                                ...                                                ...
1016  baf134d355007fbe28241a7741a19efa  �。我記得西岸上四無人煙，這番如何是好？”八戒\n道：“只說凡人會作弊，原來這佛面前的金剛也...      1200  [6bb959359e54493abf6f13abb302b826]                                               None
None
1017  145714134e48a5b11f93e8ef84519676  跟至朝門之外。\n　　\n唐僧下馬，同眾進朝。唐僧將龍馬與經擔，同行者、八戒、沙僧，站在玉階...      1200  [6bb959359e54493abf6f13abb302b826]                                               None
None
1018  090d467a2a7a7cf8bbbe97103da7f3a5  子，因有罪，也蒙菩薩救解，\n教他與臣作腳力。當時變作原馬，毛片相同。幸虧他登山越嶺，跋涉崎...      1200  [6bb959359e54493abf6f13abb302b826]                                               None
None
1019  3cb26aabe2111fc02ac274dd0767ef7d  光佛。南無大通光佛。南無才光佛。南無旃檀功德佛。南無\n鬥戰勝佛。南無觀世音菩薩。南無大勢至...      1200  [6bb959359e54493abf6f13abb302b826]                                               None
None
1020  a7cff3d70833c1df766fc6e0bb24e39f   be bound by the terms of this agreement. Ther...      1200  [6bb959359e54493abf6f13abb302b826]                                               None                                               None

[1021 rows x 6 columns]

🚀 create_base_documents
                                 id                                         text_units                                        raw_content     title
0  6bb959359e54493abf6f13abb302b826  [1e17281c57c89ae5df4e8cc31a87dfdf, e8de3e9c8ad...  The Project Gutenberg eBook of 西遊記\n    \nThi...  book.txt
🚀 create_final_documents
                                 id                                      text_unit_ids                                        raw_content     title
0  6bb959359e54493abf6f13abb302b826  [1e17281c57c89ae5df4e8cc31a87dfdf, e8de3e9c8ad...  The Project Gutenberg eBook of 西遊記\n    \nThi...  book.txt
⠼ GraphRAG Indexer
├── Loading Input (InputFileType.text) - 1 files loaded (0 filtered) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 0:00:00
├── create_base_text_units
├── create_base_extracted_entities
├── create_summarized_entities
├── create_base_entity_graph
├── create_final_entities
├── create_final_nodes
├── create_final_communities
├── join_text_units_to_entity_ids
├── create_final_relationships
├── join_text_units_to_relationship_ids
├── create_final_community_reports
├── create_final_text_units
├── create_base_documents
└── create_final_documents
🚀 All workflows completed successfully.
```

### 六、测试

```bash
python -m graphrag.query --root ./xiyouji --method local “白骨精分别变化为什么角色来诱骗唐僧？“

INFO: Vector Store Args: {}
creating llm client with {'api_key': 'REDACTED,len=9', 'type': "openai_chat", 'model': 'qwen2-7b-instruct-q5_k_m', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'n': 1, 'request_timeout': 180.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}
creating embedding llm client with {'api_key': 'REDACTED,len=9', 'type': "openai_embedding", 'model': 'qwen2-7b-instruct-q5_k_m', 'max_tokens': 4000, 'temperature': 0, 'top_p': 1, 'n': 1, 'request_timeout': 180.0, 'api_base': 'http://localhost:11434/api', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': None, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}
self.model = qwen2-7b-instruct-q5_k_m

SUCCESS: Local Search Response:
在《西游记》中，白骨精为了诱骗唐僧并吃掉他以获取长生不老之术，三次变化成不同的身份试图接近。以下是她所变的角色：

1. **第一次变化**：白骨精首先变成了一位年轻美貌的村姑，手提一篮子新鲜的野果，目的是吸引唐僧一行人停下休息，并趁机吃掉他们。

2. **第二次变化**：在被孙悟空识破并打死她的化身后，她再次变化，这次变成了一个老妇人，声称是来找女儿和丈夫（即前两次变化的身份）的。她试图通过哀求和解释来迷惑唐僧，但最终还是被孙悟空发现并消灭。

3. **第三次变化**：在最后的努力中，白骨精化身为一位年迈的老翁，继续寻找机会接近唐僧一行人，意图再次诱骗他们。然而，她的诡计仍然没有得逞，最终也被孙悟空识破并打死。

这些变化展示了白骨精的狡猾和对长生不老的渴望，同时也体现了孙悟空的智慧与警惕性。
```

```bash
python -m graphrag.query --root ./xiyouji --method global “白骨精分别变化为什么角色来诱骗唐僧？“

creating llm client with {'api_key': 'REDACTED,len=9', 'type': "openai_chat", 'model': 'qwen2-7b-instruct-q5_k_m', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'n': 1, 'request_timeout': 180.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}

SUCCESS: Global Search Response:
根据分析，白骨精首先变化为一个美丽的村姑，试图吸引唐僧的注意并诱骗他。随后，她再次变化为一位老妇人来找她的女儿（即之前的村姑），继续尝试接近唐僧。接着，她又变化为一位年轻女子，声称是她的女儿来找寻母亲，并继续尝试接近唐僧。

在后续的过程中，白骨精还变换为其他角色以诱骗唐僧。例如，在某些情况下，她化身为老妇人来找她的丈夫和女儿；在另一些情况下，她再次变化为一个年轻女子，声称是她的女儿来找寻母亲。此外，她甚至尝试过变成一只狗来逃脱被发现的命运。

这些变化显示了白骨精利用人类形态的能力以及其对目标的执着与欺骗手段的多样性。通过变换角色，她试图混淆唐僧对善恶的判断，并最终完成其诱骗计划。
```