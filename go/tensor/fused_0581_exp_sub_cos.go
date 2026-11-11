package tensor

import "fmt"

func Fused0581ExpSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Cos()
	return r, nil
}
func Fused0581ExpSubCosName() string { return "exp_sub_cos" }
