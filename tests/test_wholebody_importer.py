import pytest
from dextrah_lab.wholebody.importer import mimic_flag_from_source


def test_straight_through_and_inverted_importer_semantics():
    prefix='def configure(self):\n    import_config.set_parse_mimic('
    field='self.cfg.convert_mimic_joints_to_normal_joints'
    assert mimic_flag_from_source(prefix+field+')') is True
    assert mimic_flag_from_source(prefix+'not '+field+')') is False
    with pytest.raises(ValueError):
        mimic_flag_from_source(prefix+'False)')
    with pytest.raises(ValueError):
        mimic_flag_from_source('def configure(self): pass')
