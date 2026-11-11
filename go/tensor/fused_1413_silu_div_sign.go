package tensor

import "fmt"

func Fused1413SiluDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Sign()
	return r, nil
}
func Fused1413SiluDivSignName() string { return "silu_div_sign" }
