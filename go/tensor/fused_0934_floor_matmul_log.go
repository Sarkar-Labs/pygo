package tensor

import "fmt"

func Fused0934FloorMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Log()
	return r, nil
}
func Fused0934FloorMatmulLogName() string { return "floor_matmul_log" }
