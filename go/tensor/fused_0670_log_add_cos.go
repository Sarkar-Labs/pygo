package tensor

import "fmt"

func Fused0670LogAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Cos()
	return r, nil
}
func Fused0670LogAddCosName() string { return "log_add_cos" }
