package tensor

import "fmt"

func Fused0788SinDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.GELU()
	return r, nil
}
func Fused0788SinDivGeluName() string { return "sin_div_gelu" }
