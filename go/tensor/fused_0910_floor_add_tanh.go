package tensor

import "fmt"

func Fused0910FloorAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Tanh()
	return r, nil
}
func Fused0910FloorAddTanhName() string { return "floor_add_tanh" }
