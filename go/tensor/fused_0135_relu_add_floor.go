package tensor

import "fmt"

func Fused0135ReluAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Floor()
	return r, nil
}
func Fused0135ReluAddFloorName() string { return "relu_add_floor" }
