package tensor

import "fmt"

func Fused0769SinMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Floor()
	return r, nil
}
func Fused0769SinMatmulFloorName() string { return "sin_matmul_floor" }
