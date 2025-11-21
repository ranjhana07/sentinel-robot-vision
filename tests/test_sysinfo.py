from sentinel.sysinfo import cpu_name, environment_info


def test_cpu_name_is_not_empty():
    assert cpu_name()


def test_environment_info_has_core_fields():
    info = environment_info()
    for key in ("cpu", "logical_cores", "os", "python", "onnxruntime"):
        assert key in info
