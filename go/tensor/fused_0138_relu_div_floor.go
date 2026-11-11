package tensor

import "fmt"

func Fused0138ReluDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Floor()
	return r, nil
}
func Fused0138ReluDivFloorName() string { return "relu_div_floor" }
