/*******************************************************************************
* Copyright 2014-2016 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/** A very simple memory pool to minimize allocation time in memory-limited
 * Xeon Phi cards.
 * Basically, it's a circular buffer.
 * Allocation is simply done by returning the current offset and then shifting
 * the offset (and wrapping around it when necessary).
 *
 * Example:
 *
 * foo(); // internally uses pool->Allocate(), which will shift the offset
 * size_t fooEnd = pool->getTail();
 * ...
 * boo(); // internally uses pool->Allocate()
 * ...
 * size_t barBegin = pool->getTail();
 * bar();
 * ...
 * pool->setHeadOffset(fooEnd); // free those allocated before foo
 * ...
 * pool->setTailOffset(barBegin); // free those allocated after boo
 */

#ifndef SPMP_MEMORY_POOL
#define SPMP_MEMORY_POOL

#include <cstdlib>

namespace SpMP
{

class MemoryPool
{
public :
  MemoryPool();
  MemoryPool(size_t sz);
  ~MemoryPool();

  void initialize(size_t sz);
  void finalize();

  void setHead(size_t offset);
  void setTail(size_t offset);

  size_t getHead() const;
  size_t getTail() const;

  void *allocate(size_t sz, int align = 64);
  void *allocateFront(size_t sz, int align = 64);
  void deallocateAll();

  template<typename T> T *allocate(size_t cnt) {
    return (T *)allocate(sizeof(T)*cnt);
  }
  template<typename T> T *allocateFront(size_t cnt) {
    return (T *)allocateFront(sizeof(T)*cnt);
  }

  /**
   * @return true if ptr is in this pool
   */
  bool contains(const void *ptr) const;
  
  static MemoryPool *getSingleton();

private :
  size_t size_;
  size_t head_, tail_;
  char *buffer_;
};

} // namespace SpMP

#endif
