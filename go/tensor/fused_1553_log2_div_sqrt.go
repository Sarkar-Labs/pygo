package tensor

import "fmt"

func Fused1553Log2DivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Sqrt()
	return r, nil
}
func Fused1553Log2DivSqrtName() string { return "log2_div_sqrt" }
