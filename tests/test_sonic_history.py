import pytest
import torch
from dextrah_lab.wholebody.sonic import SonicHistory, load_export_config


def test_history_order_and_partial_reset():
    history=SonicHistory(2)
    terms=[torch.full((2,d),float(i)) for i,d in enumerate(history.DIMENSIONS)]
    first=history.push(*terms)
    assert first.shape==(2,930)
    torch.testing.assert_close(first[:,:30],torch.zeros(2,30))
    second=history.push(*[t+10 for t in terms])
    assert second[0,26]==0 and second[0,27]==10
    history.reset(torch.tensor([1]))
    third=history.push(*[t+20 for t in terms])
    assert third[0,0]==0
    assert torch.equal(third[1,:30],torch.full((30,),20.))
    with pytest.raises(ValueError):
        history.push(*terms[:-1])


def test_config_only_accepts_known_metadata_path_tag(tmp_path):
    path=tmp_path/'config.yaml'
    path.write_text('path: !!python/object/apply:pathlib.PosixPath [models, sonic]\n')
    assert load_export_config(path).path=='models/sonic'
    path.write_text('path: !!python/object/apply:os.system [false]\n')
    with pytest.raises(Exception):
        load_export_config(path)
