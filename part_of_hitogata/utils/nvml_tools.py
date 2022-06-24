import py3nvml.nvidia_smi as smi

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def get_gpu_temp(gpu_ids):
    res = dict(line1=dict())
    root = ET.fromstring(smi.XmlDeviceQuery())
    for i, gpu in enumerate(root.findall('gpu')):
        if i in gpu_ids:
            res['line1']['gpu_' + str(i) + '_temp'] = (int(gpu.find('temperature').find('gpu_temp').text.split(' ')[0]),
                                                       dict(title='gpu_' + str(i) + '_temp (C)'), True)
    return res


def get_fan_speed(gpu_ids):
    res = dict(line1=dict())
    root = ET.fromstring(smi.XmlDeviceQuery())
    for i, gpu in enumerate(root.findall('gpu')):
        if i in gpu_ids:
            res['line1']['gpu_' + str(i) + '_fan_speed'] = (int(gpu.find('fan_speed').text.split(' ')[0]),
                                                            dict(title='gpu_' + str(i) + '_fan_speed (%)'), True)
    return res


def get_utilization(gpu_ids):
    res = dict(line1=dict())
    root = ET.fromstring(smi.XmlDeviceQuery())
    for i, gpu in enumerate(root.findall('gpu')):
        if i in gpu_ids:
            res['line1']['gpu_' + str(i) + '_gpu_util'] = (int(gpu.find('utilization').find('gpu_util').
                                                               text.split(' ')[0]),
                                                           dict(title='gpu_' + str(i) + '_gpu_util (%'))
            res['line1']['gpu_' + str(i) + '_memory_util'] = (int(gpu.find('utilization').find('memory_util').
                                                                  text.split(' ')[0]),
                                                              dict(title='gpu_' + str(i) + '_memory_util (%'), True)
    return res


if __name__ == '__main__':
    print(get_gpu_temp([0, 1]))
    print(get_fan_speed([0, 1]))
    print(get_utilization([0, 1]))
