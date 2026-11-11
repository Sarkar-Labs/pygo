package tensor

import "fmt"

func Fused1598Log2DivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.GELU()
	return r, nil
}
func Fused1598Log2DivGeluName() string { return "log2_div_gelu" }
