package tensor

import "fmt"

func Fused0226SigmoidSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Floor()
	return r, nil
}
func Fused0226SigmoidSubFloorName() string { return "sigmoid_sub_floor" }
