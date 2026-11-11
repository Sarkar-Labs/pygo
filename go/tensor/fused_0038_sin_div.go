package tensor

import "fmt"

func Fused0038SinDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	return r, nil
}
func Fused0038SinDivName() string { return "sin_div" }
