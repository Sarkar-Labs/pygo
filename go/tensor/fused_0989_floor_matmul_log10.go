package tensor

import "fmt"

func Fused0989FloorMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Log10()
	return r, nil
}
func Fused0989FloorMatmulLog10Name() string { return "floor_matmul_log10" }
