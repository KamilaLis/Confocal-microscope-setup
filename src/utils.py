import xml.etree.ElementTree as Xet
import pandas as pd
import ast


def read_maf(inputFile: str) -> pd.DataFrame:
	xmlparse = Xet.parse(inputFile)
	xyzStagePointDefinitionList = xmlparse.getroot()
	rows = []
	for xyzStagePointDefinition in xyzStagePointDefinitionList:
		rows.append({"PosX": int(xyzStagePointDefinition.attrib['PosX']),
					"PosY": int(xyzStagePointDefinition.attrib['PosY']),
					"PosZ": int(xyzStagePointDefinition.attrib['PosZ']),
					"ZMode": int(xyzStagePointDefinition.attrib['ZMode']),
					"Begin": int(xyzStagePointDefinition.attrib['Begin']),
					"End": int(xyzStagePointDefinition.attrib['End']),
					"Sections": int(xyzStagePointDefinition.attrib['Sections']),
					"StepSize": int(xyzStagePointDefinition.attrib['StepSize']),
					"CycleCount": int(xyzStagePointDefinition.attrib['CycleCount']),
					"CycleTime": int(xyzStagePointDefinition.attrib['CycleTime']),
					"WaitTime": int(xyzStagePointDefinition.attrib['WaitTime']),
					"ValidStack": int(xyzStagePointDefinition.attrib['ValidStack']),
					"PositionIdentifier": xyzStagePointDefinition.attrib['PositionIdentifier'],
					"FileNameBase": xyzStagePointDefinition.attrib['FileNameBase'],
					"MartixIdentifier": int(xyzStagePointDefinition.attrib['MartixIdentifier']),	#correcting typo 'Martix' -> 'Matrix'
					"TileScanIdentifier": int(xyzStagePointDefinition.attrib['TileScanIdentifier']),
					"AFCOffset": int(xyzStagePointDefinition.attrib['AFCOffset']),
					"Valid": int(xyzStagePointDefinition[0][0].attrib['Valid']),
					"SuperZMode": int(xyzStagePointDefinition[0][0].attrib['SuperZMode']),
					"ZPosition": float(xyzStagePointDefinition[0][0].attrib['ZPosition'])
					})
	df = pd.DataFrame(rows, columns=rows[0].keys())
	return df


def read_xml(xml_file: str) -> dict:
	xmlparse = Xet.parse(xml_file)
	metadata = xmlparse.getroot()
	# return {elem.tag: elem.text for elem in metadata}

	data_dict = {}
	for elem in metadata:
		try:
			data_dict[elem.tag] = ast.literal_eval(elem.text)
		except ValueError:
			data_dict[elem.tag] = elem.text
	return data_dict


def save_maf(df: pd.DataFrame, outputFile: str) -> None:
	root = Xet.Element("XYZStagePointDefinitionList")
	for _, row in df.iterrows():
		pointDefinition = Xet.SubElement(root, "XYZStagePointDefinition", attrib={
			"PosX": str(row["PosX"]),
			"PosY": str(row['PosY']),
			"PosZ": str(row['PosZ']),
			"ZMode": str(row['ZMode']),
			"Begin": str(row['Begin']),
			"End": str(row['End']),
			"Sections": str(row['Sections']),
			"StepSize": str(row['StepSize']),
			"CycleCount": str(row['CycleCount']),
			"CycleTime": str(row['CycleTime']),
			"WaitTime": str(row['WaitTime']),
			"ValidStack": str(row['ValidStack']),
			"PositionIdentifier": row['PositionIdentifier'],
			"FileNameBase": row['FileNameBase'],
			"MartixIdentifier": str(row['MartixIdentifier']),
			"TileScanIdentifier": str(row['TileScanIdentifier']),
			"AFCOffset": str(row['AFCOffset']),
		})
		zPositionList = Xet.SubElement(pointDefinition, "AdditionalZPositionList")
		Xet.SubElement(zPositionList, "AdditionalZPosition", attrib={
			"Valid": str(row['Valid']),
			"SuperZMode": str(row['SuperZMode']),
			"ZMode": str(row['ZMode']),
			"ZPosition": str(row['ZPosition'])
		})

	tree = Xet.ElementTree(root)
	Xet.indent(tree)
	tree.write(outputFile, xml_declaration=True)
