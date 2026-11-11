package tensor

import "fmt"

func Fused0225SigmoidAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Floor()
	return r, nil
}
func Fused0225SigmoidAddFloorName() string { return "sigmoid_add_floor" }
