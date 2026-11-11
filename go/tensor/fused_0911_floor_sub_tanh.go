package tensor

import "fmt"

func Fused0911FloorSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Tanh()
	return r, nil
}
func Fused0911FloorSubTanhName() string { return "floor_sub_tanh" }
