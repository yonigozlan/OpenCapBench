import yaml

def importMetadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)

    return parsedYamlFile

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
