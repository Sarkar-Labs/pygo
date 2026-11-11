package tensor

import "fmt"

func Fused0836CosSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Exp()
	return r, nil
}
func Fused0836CosSubExpName() string { return "cos_sub_exp" }
