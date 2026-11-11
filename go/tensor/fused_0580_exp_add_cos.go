package tensor

import "fmt"

func Fused0580ExpAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Cos()
	return r, nil
}
func Fused0580ExpAddCosName() string { return "exp_add_cos" }
