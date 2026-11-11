package tensor

import "fmt"

func Fused1074CeilMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Log2()
	return r, nil
}
func Fused1074CeilMatmulLog2Name() string { return "ceil_matmul_log2" }
