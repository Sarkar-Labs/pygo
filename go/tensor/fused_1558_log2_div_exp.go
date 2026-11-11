package tensor

import "fmt"

func Fused1558Log2DivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Exp()
	return r, nil
}
func Fused1558Log2DivExpName() string { return "log2_div_exp" }
