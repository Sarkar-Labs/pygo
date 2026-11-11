package tensor

import "fmt"

func Fused0229SigmoidMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Floor()
	return r, nil
}
func Fused0229SigmoidMatmulFloorName() string { return "sigmoid_matmul_floor" }
