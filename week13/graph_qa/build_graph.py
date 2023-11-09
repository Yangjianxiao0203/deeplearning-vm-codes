import re
import json
from py2neo import Graph
from collections import defaultdict
from config import Config

def connect_graph(config):
    url = config["graph"]["url"]
    username = config["graph"]["username"]
    password = config["graph"]["password"]
    graph = Graph(url, auth=(username, password))
    return graph

def read_file(file_path,attr_dict,label_dict):
    with open(file_path, encoding="utf8") as f:
        for line in f:
            head, relation, tail = line.strip().split("\t")
            head = get_label_then_clean(head, label_dict)
            attr_dict[head][relation] = tail
    return

def get_label_then_clean(x, label_data):
    if re.search("（.+）", x):
        label_string = re.search("（.+）", x).group()
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)  # 括号内的内容删掉，因为括号是特殊字符会影响cypher语句运行
                label_data[x] = label
            else:
                x = re.sub("（.+）", "", x)
    return x

def load_dicts(config):
    #1加载文件，把三元组的元素读到对应的字典, key 都是entity,除了label和实体是一对一，其他attr和relation都是一个实体对多个
    #e.g attr_dict = {"entity1":{"attr1":"value1","attr2":"value2"},"entity2":{"attr1":"value1","attr2":"value2"}}
    #e.g label_dict = {"entity1":"label1","entity2":"label2"}
    label_dict = {}
    attr_dict = defaultdict(dict)
    relation_dict = defaultdict(dict)
    dicts = [attr_dict,relation_dict,label_dict]
    files = config["files"]
    for file,_dict in zip(files,dicts):
        read_file(file,_dict,label_dict)
    return dicts

def build_cypher(attr_dict,relation_dict,label_dict):
    cypher = ""
    in_graph_entity = set()
    # Create语句： CREATE (n:label {attr1:value1,attr2:value2})
    # 主意，有几个create语句，就会创建几个节点，如果有相同的节点，会创建多个

    # attr_cypher: node的创建
    for entity in attr_dict:
        # add a name attribute for all entities
        attr_dict[entity]["NAME"] = entity
        text = "{"
        for attr, value in attr_dict[entity].items():
            text += "%s:\'%s\'," % (attr, value) #这里的value是字符串，所以要加上单引号, \'转义字符，相当于', 用来确保插入的单引号不会被解释为字符串字面量的结束符号
        text = text[:-1] + "}" #去掉最后一个逗号
        if entity in label_dict:
            label = label_dict[entity]
            cypher += "CREATE (%s:%s %s)\n" % (entity,label, text)
        else:
            cypher += "CREATE (%s %s)\n" % (entity, text)
        in_graph_entity.add(entity)

    # relation_cypher: edge的创建
    # relation_dict = {"entity1":{"relation1":"entity2","relation2":"entity3"}}
    # match查询语句：MATCH (n1:label1 {attr1:value1})-[r:relation]->(n2:label2 {attr2:value2})
    # create语句：CREATE (n1)-[r:relation]->(n2)
    for e1 in relation_dict:
        for relation, e2 in relation_dict[e1].items():
            if e1 not in in_graph_entity:
                cypher += "CREATE (%s {NAME:'%s'})\n" % (e1, e1) # add a name attribute
                in_graph_entity.add(e1)
            if e2 not in in_graph_entity:
                cypher += "CREATE (%s {NAME:'%s'})\n" % (e2, e2)
                in_graph_entity.add(e2)
            cypher += "CREATE (%s)-[:%s]->(%s)\n" % (e1, relation, e2)

    return cypher

def store_dicts(dicts,config):
    attr_dict,relation_dict,label_dict = dicts
    results = defaultdict(set)
    for entity in attr_dict:
        results["entitys"].add(entity)
        for attr,value in attr_dict[entity].items():
            results["attributes"].add(attr)
            results["values"].add(value)
    for entity in relation_dict:
        results["entitys"].add(entity)
        for relation,value in relation_dict[entity].items():
            results["relations"].add(relation)
            results["entitys"].add(value)
    for entity in label_dict:
        results["entitys"].add(entity)
        results["labels"].add(label_dict[entity])
    results = {k:list(v) for k,v in results.items()}
    with open(config["schema"], "w", encoding="utf8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
    return results

def build(config,clean=True):
    # 1.加载文件
    dicts = load_dicts(config)
    #2.构建cypher语句
    cypher = build_cypher(*dicts)
    print("*"*50 +"\n"+ cypher + "*"*50)
    #3.连接图数据库
    graph = connect_graph(config)
    #4.执行cypher语句
    if clean:
        graph.run("MATCH (n) DETACH DELETE n")
    graph.run(cypher)
    # store the dicts
    store_dicts(dicts,config)

if __name__ == "__main__":
    build(Config)
