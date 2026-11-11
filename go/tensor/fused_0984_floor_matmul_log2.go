package tensor

import "fmt"

func Fused0984FloorMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Log2()
	return r, nil
}
func Fused0984FloorMatmulLog2Name() string { return "floor_matmul_log2" }
