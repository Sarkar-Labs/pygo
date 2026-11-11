package tensor

import "fmt"

func Fused1573Log2DivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Cos()
	return r, nil
}
func Fused1573Log2DivCosName() string { return "log2_div_cos" }
