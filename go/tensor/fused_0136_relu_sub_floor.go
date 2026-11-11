package tensor

import "fmt"

func Fused0136ReluSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Floor()
	return r, nil
}
func Fused0136ReluSubFloorName() string { return "relu_sub_floor" }
