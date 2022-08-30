import onnx

onnx_model = onnx.load('supercombo.onnx')

endpoint_names = ['input_imgs', 'big_input_imgs', 'desire', 'traffic_convention', 'initial_state', 'outputs']

for i in range(len(onnx_model.graph.node)):
    for j in range(len(onnx_model.graph.node[i].input)):
        if onnx_model.graph.node[i].input[j] in endpoint_names:
            print('-' * 60)
            print(onnx_model.graph.node[i].name)
            print(onnx_model.graph.node[i].input)
            print(onnx_model.graph.node[i].output)

            onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].split(':')[0]

    for j in range(len(onnx_model.graph.node[i].output)):
        if onnx_model.graph.node[i].output[j] in endpoint_names:
            print('-' * 60)
            print(onnx_model.graph.node[i].name)
            print(onnx_model.graph.node[i].input)
            print(onnx_model.graph.node[i].output)

            onnx_model.graph.node[i].output[j] = onnx_model.graph.node[i].output[j].split(':')[0]

for i in range(len(onnx_model.graph.input)):
    if onnx_model.graph.input[i].name in endpoint_names:
        print('-' * 60)
        print(onnx_model.graph.input[i])
        onnx_model.graph.input[i].name = onnx_model.graph.input[i].name.split(':')[0]

for i in range(len(onnx_model.graph.output)):
    if onnx_model.graph.output[i].name in endpoint_names:
        print('-' * 60)
        print(onnx_model.graph.output[i])
        onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.split(':')[0]

onnx.save(onnx_model, 'model.onnx')