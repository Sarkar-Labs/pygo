package tensor

import "fmt"

func Fused0710LogAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Log2()
	return r, nil
}
func Fused0710LogAddLog2Name() string { return "log_add_log2" }
