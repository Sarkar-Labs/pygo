package tensor

import "fmt"

func Fused0139ReluMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Floor()
	return r, nil
}
func Fused0139ReluMatmulFloorName() string { return "relu_matmul_floor" }
