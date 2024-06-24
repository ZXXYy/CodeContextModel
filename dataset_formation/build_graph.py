from doxmlparser import DoxygenType, CompoundType
import doxmlparser

# process a compound file and export the results to stdout
def parse_compound(inDirName,baseName):
    doxmlparser.compound.parse(inDirName+"/"+baseName+".xml",False)

# process the index file and export the results to stdout
def parse_index(inDirName):
    rootObj = doxmlparser.index.parse(inDirName+"/index.xml",False)
    for compound in rootObj.get_compound(): # for each compound defined in the index
        parse_compound(inDirName,compound.get_refid())
# # 解析 index.xml 文件
# parse_classes("xml/index.xml")

parse_index('/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/structgraph/xml')