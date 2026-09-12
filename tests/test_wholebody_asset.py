import xml.etree.ElementTree as ET
import pytest
from dextrah_lab.wholebody.asset import audit_urdf, prepare_urdf
from dextrah_lab.wholebody.contract import BODY_JOINTS, hand_joints


def fixture(tmp_path):
    root=ET.Element('robot',name='test')
    names=[*BODY_JOINTS,*hand_joints('left'),*hand_joints('right')]
    for side in ('left','right'):
        for finger in ('thumb','index','middle','ring','pinky'):
            names.append(f'{side}_{finger}_distal_joint')
    for name in ['pelvis',*names,'sensor']:
        link=ET.SubElement(root,'link',name=name)
        inertial=ET.SubElement(link,'inertial')
        ET.SubElement(inertial,'mass',value='1')
        ET.SubElement(inertial,'inertia',ixx='1',iyy='1',izz='1',ixy='0',ixz='0',iyz='0')
        geom=ET.SubElement(ET.SubElement(link,'collision'),'geometry')
        ET.SubElement(geom,'box',size='.01 .01 .01')
        if name=='pelvis':
            continue
        joint=ET.SubElement(root,'joint',name=name,type='revolute')
        ET.SubElement(joint,'parent',link='pelvis')
        ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'axis',xyz='0 0 1')
        ET.SubElement(joint,'limit',lower='0',upper='0' if name=='sensor' else '1.5',effort='1',velocity='1')
        if '_distal_' in name:
            ET.SubElement(joint,'mimic',joint=name.replace('_distal_','_proximal_'),
                          multiplier='1' if 'thumb' in name else '1.155',offset='0')
    path=tmp_path/'source.urdf'
    ET.ElementTree(root).write(path)
    return path


def test_full_asset_preserves_branches_mass_couplings_and_geometry(tmp_path):
    source=fixture(tmp_path)
    before=audit_urdf(source)
    result=prepare_urdf(source,tmp_path/'prepared.urdf')
    after=audit_urdf(tmp_path/'prepared.urdf')
    assert result['removed_branches']==[]
    assert after['links']==before['links']
    assert after['total_mass_kg']==before['total_mass_kg']
    assert after['collision_shapes']==before['collision_shapes']
    assert len(after['mimic_relations'])==10
    assert after['zero_range_joints']==[]
    assert before['zero_range_joints']==['sensor']
    assert not after['physics_validated']
    with pytest.raises(FileExistsError):
        prepare_urdf(source,tmp_path/'prepared.urdf')


def test_reduced_robot_and_wrong_coupling_fail(tmp_path):
    source=fixture(tmp_path)
    tree=ET.parse(source)
    tree.find(".//joint[@name='left_hip_pitch_joint']").set('type','fixed')
    tree.write(source)
    with pytest.raises(ValueError,match='Required joint'):
        audit_urdf(source)
    source=fixture(tmp_path)
    tree=ET.parse(source)
    tree.find('.//mimic').set('multiplier','-1')
    tree.write(source)
    with pytest.raises(ValueError,match='coupling'):
        audit_urdf(source)


def test_bad_inertia_fails(tmp_path):
    source=fixture(tmp_path)
    tree=ET.parse(source)
    tree.find('.//inertia').set('ixx','-1')
    tree.write(source)
    with pytest.raises(ValueError,match='mass/inertia'):
        audit_urdf(source)
