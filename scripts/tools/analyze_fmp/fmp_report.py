# 输入数据
input_data = """
实验组：2,886、5,475、8,348
(pv: 7,712)
对照组：3,192、5,742、8,700
(pv: 7,877)
剩余组：3,303、6,006、8,690
(pv: 36,007)
"""


def analyze_data(input_data):
    # 解析输入数据并忽略含有pv的行
    lines = input_data.strip().split('\n')
    
    # 提取百分位数数据
    experiment_group = list(map(lambda x: float(x.replace(',', '')), lines[0].split('：')[1].split('、')))
    control_group = list(map(lambda x: float(x.replace(',', '')), lines[2].split('：')[1].split('、')))
    remaining_group = list(map(lambda x: float(x.replace(',', '')), lines[4].split('：')[1].split('、')))
    
    # 计算收益值和收益率
    results = []
    
    for i in range(3):
        experiment_value = experiment_group[i]
        control_value = control_group[i]
        
        # 计算收益值
        gain = control_value - experiment_value
        # 计算收益率
        gain_rate = (gain / control_value) * 100 if control_value != 0 else 0
        
        results.append((gain, gain_rate))
    
    # 格式化输出结果
    output = "依次为80、95、99分位\n"
    output += "收益ms："
    
    output_values = []
    for i, (gain, rate) in enumerate(results):
        sign = "+" if gain >= 0 else ""
        output_values.append(f"{int(gain)}（{sign}{rate:.1f}%）")
    
    output += "、".join(output_values)
    
    return output

# 调用函数并输出结果
result = analyze_data(input_data)
print(result)
