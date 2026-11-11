package tensor

import "fmt"

func Fused1285GeluAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Exp()
	return r, nil
}
func Fused1285GeluAddExpName() string { return "gelu_add_exp" }
