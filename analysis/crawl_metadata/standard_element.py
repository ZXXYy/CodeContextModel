"""
用于从xml记录的elements中，解析出各个元素的qualified name
如果是方法，还需要解析出参数
"""


def solve_one(element: str):
    """
    针对单个元素，解析其 qualified name
    :param element: 原始字符串
    :return: [repo, qualified name]
    """
    index1 = element.find('src&lt;')
    index2 = element.find('{')
    if index1 == -1 or index2 == -1:
        return ['', '']
    repo_str = element[1:index1 - 1]
    package_str = element[index1 + 7: index2]
    code_str = element[index2 + 1:]
    qualified_name = ''
    if code_str.endswith('.java'):
        qualified_name = package_str + '.' + code_str[:-5]
    else:
        index3 = code_str.find('.java[')
        if index3 == -1:
            return ['', '']
        qualified_name = package_str + '.'
        class_field_method_str = code_str[index3 + 6:]
        while True:
            class_index = class_field_method_str.find('[')
            if class_index == -1 or class_field_method_str[class_index-2:class_index+1] == r'~\[':
                break
            elif class_index == len(class_field_method_str) - 1:
                class_field_method_str = class_field_method_str[:class_index]
            elif class_field_method_str[class_index + 1] == '~' \
                    or class_field_method_str[class_index + 1] == '!' \
                    or class_field_method_str[class_index + 1] == '|':
                # 这个暂时不知道后面的解析情况，先直接舍弃
                class_field_method_str = class_field_method_str[:class_index]
            else:
                qualified_name += class_field_method_str[:class_index] + '.'
                class_field_method_str = class_field_method_str[class_index + 1:]
        # 到这里就没有类了
        field_method_str = class_field_method_str
        field_index = field_method_str.find('^')
        if field_index != -1:
            qualified_name += field_method_str[:field_index] + '.'
            field_str = field_method_str[field_index + 1:]
            qualified_name += field_str
        method_index = field_method_str.find('~')
        if method_index != -1:
            qualified_name += field_method_str[:method_index] + '.'
            method_str = field_method_str[method_index + 1:]
            params = method_str.split('~')
            method_str = params[0] + '('
            for p in params[1:]:
                # java描述符
                if p == "I":
                    method_str += 'int'
                elif p == r'\[I':
                    method_str += 'int[]'
                elif p == "Z":
                    method_str += 'boolean'
                elif p == r'\[Z':
                    method_str += 'boolean[]'
                elif p == r'\[B':
                    method_str += 'byte[]'
                elif p == r'B':
                    method_str += 'byte'
                elif p == r'\[C':
                    method_str += 'char[]'
                elif p == r'C':
                    method_str += 'char'
                elif p == r'\[S':
                    method_str += 'short[]'
                elif p == r'S':
                    method_str += 'short'
                elif p == r'\[F':
                    method_str += 'float[]'
                elif p == r'F':
                    method_str += 'float'
                elif p == r'\[J':
                    method_str += 'long[]'
                elif p == r'J':
                    method_str += 'long'
                elif p == r'\[D':
                    method_str += 'double[]'
                elif p == r'D':
                    method_str += 'double'
                elif p.startswith('Q'):
                    method_str += p[1:-1]
                elif p.startswith(r'\[Q'):
                    method_str += p[3:-1] + '[]'
                if params.index(p) < len(params) - 1:
                    method_str += ','
            method_str += ')'
            # 参数可能出现容器
            while True:
                less_index = method_str.find(r'\&lt;')
                great_index = method_str.find('&gt;')
                if less_index != -1 and great_index != -1:
                    sub_str = method_str[less_index: great_index+4]
                    new_sub_str = sub_str.replace('+', '')
                    new_sub_str = new_sub_str.replace(r'\&lt;', '<')
                    new_sub_str = new_sub_str.replace('&gt;', '>')
                    new_str = new_sub_str[1: -2]
                    str_list = new_str.split(';')
                    if '' in str_list:
                        str_list.remove('')
                    transfer_str = ''
                    for s in str_list:
                        transfer_str += s[1:]
                        if str_list.index(s) != len(str_list) - 1:
                            transfer_str += ','
                    new_sub_str = '<' + transfer_str + '>'
                    method_str = method_str.replace(sub_str, new_sub_str)
                else:
                    break
            qualified_name += method_str
        if field_index == -1 and method_index == -1:
            qualified_name += field_method_str
        qualified_name = qualified_name.replace('!1', '')
        qualified_name = qualified_name.replace('!2', '')
        qualified_name = qualified_name.replace('!3', '')
        qualified_name = qualified_name.replace('!4', '')
        qualified_name = qualified_name.replace('!5', '')
        qualified_name = qualified_name.replace('!6', '')
        qualified_name = qualified_name.replace('!7', '')
        qualified_name = qualified_name.replace('!8', '')
        qualified_name = qualified_name.replace('!9', '')
        qualified_name = qualified_name.replace('|1', '')
        qualified_name = qualified_name.replace('|2', '')
        qualified_name = qualified_name.replace(';!', '')
    return [repo_str, qualified_name]


def solve(elements: list[str]):
    """
    根据elements的初始文本，解析其所属的项目，文件，类，方法等
    :param elements: 原始的elements文本集合，如=org.eclipse.mylyn.tasks.tests/src&lt;org.eclipse.mylyn.tasks.tests{AllTasksTests.java
    :return: 解析之后分repo的标准的elements描述集合 如org.eclipse.mylyn.tasks.tests.AllTasksTests
    """
    standard_elements = []
    for element in elements:
        res = solve_one(element)
        standard_elements.append(res)
        # if res != '':
        #     standard_elements.append(res)
        # else:
        #     # 解析失败需要设置valid字段
        #     print('{} solve error!!!!'.format(element))
    return standard_elements
