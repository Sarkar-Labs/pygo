package tensor

import "fmt"

func Fused0068GeluDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	return r, nil
}
func Fused0068GeluDivName() string { return "gelu_div" }
