package tensor

import "fmt"

func Fused0073SiluDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	return r, nil
}
func Fused0073SiluDivName() string { return "silu_div" }
