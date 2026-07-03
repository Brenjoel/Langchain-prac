# pytest . -s -v
# . indicates current working directory
#  -s to display from stdout    
#  -v verbose flag, that shows the test we run
import pytest 

def test_foo() -> None:
    assert 1==1

def test_foo1() -> None:
    assert 1==1

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))