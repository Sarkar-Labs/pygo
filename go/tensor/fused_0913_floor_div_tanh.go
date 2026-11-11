package tensor

import "fmt"

func Fused0913FloorDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Tanh()
	return r, nil
}
func Fused0913FloorDivTanhName() string { return "floor_div_tanh" }
