package tensor

import "fmt"

func Fused1373SiluDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Sqrt()
	return r, nil
}
func Fused1373SiluDivSqrtName() string { return "silu_div_sqrt" }
