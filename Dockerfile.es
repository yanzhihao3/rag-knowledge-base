# ES + IK 中文分词器镜像。IK 插件版本必须与 ES 版本严格一致。
# 若该版本没有对应 IK 构建（安装 404），fallback 是把 es_api.py 的
# ik_max_word 换成 ES 自带的 smartcn 并重建索引。
FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.0
RUN bin/elasticsearch-plugin install --batch \
    https://github.com/infinilabs/analysis-ik/releases/download/v8.17.0/elasticsearch-analysis-ik-8.17.0.zip
