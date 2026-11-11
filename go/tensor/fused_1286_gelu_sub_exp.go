package tensor

import "fmt"

func Fused1286GeluSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Exp()
	return r, nil
}
func Fused1286GeluSubExpName() string { return "gelu_sub_exp" }
