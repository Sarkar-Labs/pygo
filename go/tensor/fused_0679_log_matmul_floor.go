package tensor

import "fmt"

func Fused0679LogMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Floor()
	return r, nil
}
func Fused0679LogMatmulFloorName() string { return "log_matmul_floor" }
