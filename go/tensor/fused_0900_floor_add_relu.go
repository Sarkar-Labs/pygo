package tensor

import "fmt"

func Fused0900FloorAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.ReLU()
	return r, nil
}
func Fused0900FloorAddReluName() string { return "floor_add_relu" }
